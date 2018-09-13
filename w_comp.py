from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms
import numpy as np
from mnist_for_labelshift import MNIST_SHIFT
from cifar10_for_labelshift import CIFAR10_SHIFT
import torchvision
import cvxpy as cp
import csv
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        self.D_in = D_in
        self.H = H
        self.D_out = D_out
        self.model = torch.nn.Sequential(
			torch.nn.Linear(self.D_in, self.H),
			torch.nn.ReLU(),
			torch.nn.Linear(self.H, self.D_out),
			)

    def forward(self, x):
    	x = x.view(-1, self.D_in)
        x = self.model(x)
        return x

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train(args, model, device, train_loader, optimizer, epoch, weight=None):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        if weight is None:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.CrossEntropyLoss(weight.float())
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    prediction = np.empty([0,1])
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            criterion = nn.CrossEntropyLoss(reduction='sum')
            loss = criterion(output, target)
            test_loss += loss.item()# sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

            correct += pred.eq(target.view_as(pred)).sum().item()
            pred = pred.numpy()
            prediction = np.concatenate((prediction, pred))

    test_loss /= len(test_loader.dataset)
    print('Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return prediction

def compute_w_inv(C_yy, mu_y):
    # compute weights

    try:
        w = np.matmul(np.linalg.inv(C_yy),  mu_y)

        print('Estimated w is', w)
        # fix w < 0
        w[np.where(w < 0)[0]] = 0
        print('If there is negative w, fix with 0:', w)
        return w
    except np.linalg.LinAlgError as err:
        if 'Singular matrix' in str(err):
            print('Cannot compute using matrix inverse due to singlar matrix')
            return np.zeros(mu_y.shape[0])*1000
        else:
            raise RuntimeError("Unknown error")
    
  

def compute_w_opt(C_yy,mu_y,mu_train_y):
    n = C_yy.shape[0]
    theta = cp.Variable(n)
    b = mu_y - mu_train_y
    objective = cp.Minimize(cp.pnorm(C_yy*theta - b) + 0.005* cp.pnorm(theta))
    constraints = [-1 <= theta, theta <= 10]
    prob = cp.Problem(objective, constraints)

    # The optimal objective value is returned by `prob.solve()`.
    result = prob.solve()
    # The optimal value for x is stored in `x.value`.
    # print(theta.value)
    w = 1 + theta.value
    print('Estimated w is', w)
    #print(constraints[0].dual_value)
    return w


def main():
    # Training settingss
    parser = argparse.ArgumentParser(description='Blackbox Label Shift')
    parser.add_argument('--iterations', type=int, default=20, metavar='N',
                        help='number of iterations to plot weight comparison plot (default: 20)')
    parser.add_argument('--data-name', type=str, default='mnist', metavar='N',
                        help='dataset name, mnist or cifar10 (default: mnist)')
    parser.add_argument('--sample-size', type=int, default=50000, metavar='N',
                        help='sample size for both training and testing (default: 50000)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs-estimation', type=int, default=10, metavar='N',
                        help='number of epochs in weight estimation (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    mse1_vec = np.ones([args.iterations, 8])
    mse2_vec = np.ones([args.iterations, 8])

    print(args.iterations)

    for k in range(args.iterations):
        for l in range(8):

            if args.data_name  == 'mnist':
                raw_data = MNIST_SHIFT('data/mnist', args.sample_size, 2, 0.4*(l+1), target_label=2, train=True, download=True,
                	transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
                D_in = 784
                
            elif args.data_name == 'cifar10':
                raw_data = CIFAR10_SHIFT('data/cifar10', args.sample_size, 2, 0.1*(l+1), target_label=2,
                    transform=transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ]), download=True)
                D_in = 3072
                #model = VGG('VGG19').to(device)#ConvNet().to(device)
            else:
                raise RuntimeError("Unsupported dataset")

            # saparate into training and testing
            m = len(raw_data)
            m_test = raw_data.get_testsize()
            print('Test size,', m_test)
            n_class = 10
            test_indices = range(m_test)
            test_data = data.Subset(raw_data, test_indices)
            train_data = data.Subset(raw_data, range(m_test, m))
            # saparate into training and validation
            m_train = m -  m_test
            m_train_t = int(m_train/2)
            print('Training_1 size,', m_train_t)

            train_t_data = data.Subset(train_data, range(m_train_t))
            train_v_data = data.Subset(train_data, range(m_train_t, m_train))

            # get labels for future use
            test_labels = raw_data.get_test_label()
            train_labels = raw_data.get_train_label()
            train_t_labels = train_labels[(range(m_train_t),)]
            train_v_labels = train_labels[(range(m_train_t, m_train),)]

            # finish data preprocessing
            # estimate weights using training and validation set
            train_loader = data.DataLoader(train_t_data,
                batch_size=args.batch_size, shuffle=True, **kwargs)

            test_loader = data.DataLoader(train_v_data,
            	batch_size=args.batch_size, shuffle=False, **kwargs)
            
            model = Net(D_in, 256, 10).to(device)
            #model = ResNet18(**kwargs).to(device)#ConvNet().to(device)
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)
            print('\nTraining using training_data1, testing on training_data2 to estimate weights.') 
            for epoch in range(1, args.epochs_estimation + 1):
                train(args, model, device, train_loader, optimizer, epoch)
            
            print('\nTesting on training_data2 to estimate C_yy.')
            predictions = test(args, model, device, test_loader)

            # compute C_yy 
            #predictions = torch.tensor(predictions)
            C_yy = np.zeros((n_class, n_class))
            m_train_v = m_train - m_train_t 
            #print(m_train_v)
            predictions = np.concatenate(predictions)
           
            for i in range(n_class):
                for j in range(n_class):
                    C_yy[i,j] = float(len(np.where((predictions== i)&(train_v_labels==j))[0]))/m_train_v
        		
            mu_y_train = np.zeros(n_class)
            for i in range(n_class):
                mu_y_train[i] = float(len(np.where(predictions == i)[0]))/m_train_v

            # print(mu_y_train)

        	#print(C_yy)
        	# prediction on x_test to estimate mu_y
            print('\nTesting on test data to estimate mu_y.')
            test_loader = data.DataLoader(test_data,
            	batch_size=args.batch_size, shuffle=False, **kwargs)
            predictions = test(args, model, device, test_loader)
            mu_y = np.zeros(n_class)
            for i in range(n_class):
                mu_y[i] = float(len(np.where(predictions == i)[0]))/m_test

            # print(mu_y)

            w1 = compute_w_inv(C_yy, mu_y)
            w2 = compute_w_opt(C_yy, mu_y, mu_y_train)

            # compute the true w
            mu_y_train = np.zeros(n_class)
            for i in range(n_class):
                mu_y_train[i] = float(len(np.where(train_v_labels == i)[0]))/m_train_v
            mu_y_test = np.zeros(n_class)
            for i in range(n_class):
                mu_y_test[i] = float(len(np.where(test_labels == i)[0]))/m_test
            true_w = mu_y_test/mu_y_train
            print('True w is', true_w)
            mse1 = sum(np.square(true_w - w1))/n_class
            mse2 = sum(np.square(true_w - w2))/n_class
            print('Mean square error, ', mse1)
            print('Mean square error, ', mse2)


            mse1_vec[k,l] = mse1
            mse2_vec[k,l] = mse2



    #######write to files#########3
    np.savetxt("mse1.csv", mse1_vec, delimiter=",")
    np.savetxt("mse2.csv", mse2_vec, delimiter=",")



if __name__ == '__main__':
    main()









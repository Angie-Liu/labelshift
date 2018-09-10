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
from mnist_for_labelshift import WEIGHTED_DATA

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.D_in = 784
        self.H = 256
        self.D_out = 10
        self.model = torch.nn.Sequential(
			torch.nn.Linear(self.D_in, self.H),
			torch.nn.ReLU(),
			torch.nn.Linear(self.H, self.D_out),
			)

    def forward(self, x):
    	x = x.view(-1, 28*28)
        x = self.model(x)
        x = F.log_softmax(x, dim=1)
        return x
        

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
   
        
        loss = F.nll_loss(output, target)
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
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

            correct += pred.eq(target.view_as(pred)).sum().item()
            pred = pred.numpy()
            prediction = np.concatenate((prediction, pred))

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return prediction




def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    mnist_data = MNIST_SHIFT('../data', 10000, 3, 0.1, target_label=2, train=True, download=False,
    	transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                       ]))

    # saparate into training and testing
    # the first 20000 are testing set
    m = len(mnist_data)
    print(m)
    m_test = mnist_data.get_testsize()
    print(m_test)
    n_class = 10
    test_indices = range(m_test)
    test_data = data.Subset(mnist_data, test_indices)
    train_data = data.Subset(mnist_data, range(m_test, m))
    # saparate into training and validation
    m_train = m -  m_test
    m_train_t = int(m_train/2)
    print(m_train_t)

    train_t_data = data.Subset(train_data, range(m_train_t))
    train_v_data = data.Subset(train_data, range(m_train_t, m_train))

    # get labels for future use
    test_labels = mnist_data.get_test_label()
    train_labels = mnist_data.get_train_label()
    train_t_labels = train_labels[(range(m_train_t),)]
    train_v_labels = train_labels[(range(m_train_t, m_train),)]

    # finish data preprocessing
    # estimate weights using training and validation set
    train_loader = data.DataLoader(train_t_data,
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = data.DataLoader(train_v_data,
    	batch_size=args.batch_size, shuffle=False, **kwargs)
    

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
    
    predictions = test(args, model, device, test_loader)

    # compute C_yy
    
    #predictions = torch.tensor(predictions)
    C_yy = np.zeros((n_class, n_class))
    m_train_v = m_train - m_train_t 
    print(m_train_v)
    
    predictions = np.concatenate(predictions)
    train_v_labels = train_v_labels.numpy()
   

    for i in range(n_class):
        for j in range(n_class):
            C_yy[i,j] = float(len(np.where((predictions== i)&(train_v_labels==j))[0]))/m_train_v
		
	print(C_yy)
	# prediction on x_test to estimate mu_y
    test_loader = data.DataLoader(test_data,
    	batch_size=args.batch_size, shuffle=False, **kwargs)
    predictions = test(args, model, device, test_loader)

    mu_y = np.zeros(n_class)
    for i in range(n_class):
        mu_y[i] = float(len(np.where(predictions == i)[0]))/m_test

    print(mu_y)
	# compute weights
    print(np.linalg.inv(C_yy*10))
    w = np.matmul(np.linalg.inv(C_yy),  mu_y)

    print('w is', w)

    # compute the true w
    mu_y_train = np.zeros(n_class)
    for i in range(n_class):
        mu_y_train[i] = float(len(np.where(train_v_labels == i)[0]))/m_train_v

    mu_y_test = np.zeros(n_class)
    for i in range(n_class):
        mu_y_test[i] = float(len(np.where(test_labels == i)[0]))/m_test

    true_w = mu_y_test/mu_y_train

    print('true w is', true_w)

    mse = sum(np.square(true_w - w))/n_class

    print('mean square error, ', mse)

	# Learning IW ERM
    weights = [w[train_labels]][0]

    weighted_train = WEIGHTED_DATA(train_data, weights)

    train_loader = data.DataLoader(weighted_train,
        batch_size=args.batch_size, shuffle=True, **kwargs)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
    
    test(args, model, device, test_loader)







if __name__ == '__main__':
    main()



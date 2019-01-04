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
from resnet import *
import cvxpy as cp
# from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
import os
import copy

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
            criterion = nn.CrossEntropyLoss(weight)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader, weight=None):
    model.eval()
    test_loss = 0
    correct = 0
    prediction = np.empty([0,1])
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            if weight is None:
                criterion = nn.CrossEntropyLoss(reduction='sum')
            else:
                criterion = nn.CrossEntropyLoss(weight, reduction='sum')

            loss = criterion(output, target)
            test_loss += loss.item()# sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

            correct += pred.eq(target.view_as(pred)).sum().item()
            pred = pred.cpu().numpy()
            prediction = np.concatenate((prediction, pred))

    test_loss /= len(test_loader.dataset)
    print('Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return prediction, 100. * correct / len(test_loader.dataset), test_loss


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
            print('Cannot compute using matrix inverse due to singlar matrix, using psudo inverse')
            w = np.matmul(np.linalg.pinv(C_yy), mu_y)
            w[np.where(w < 0)[0]] = 0
            return w
        else:
            raise RuntimeError("Unknown error")
    

def compute_w_opt(C_yy,mu_y,mu_train_y, rho):
    n = C_yy.shape[0]
    theta = cp.Variable(n)
    b = mu_y - mu_train_y
    objective = cp.Minimize(cp.pnorm(C_yy*theta - b) + rho* cp.pnorm(theta))
    constraints = [-1 <= theta]
    prob = cp.Problem(objective, constraints)

    result = prob.solve()
    w = 1 + theta.value 
    print('Estimated w is', w)
   
    return w

def compute_3deltaC(n_class, n_train, delta):
    rho = 3*(2*np.log(2*n_class/delta)/(3*n_train) + np.sqrt(2*np.log(2*n_class/delta)/n_train))
    return rho 

def choose_alpha(n_class, C_yy, mu_y, mu_y_train, rho, true_w):
    alpha = [10, 1, 0.1, 0.01, 0.001, 0.0001]
    w2 = np.zeros((len(alpha), n_class))
    for i in range(len(alpha)):

        w2[i, :] = compute_w_opt(C_yy, mu_y, mu_y_train, alpha[i] * rho)
    mse2 = np.sum(np.square(np.matlib.repmat(true_w, len(alpha),1) - w2), 1)/n_class
    i = np.argmin(mse2)
    print("mse2, ", mse2)
    return alpha[i]


def compute_true_w(train_labels, test_labels, n_class, m_train, m_test):
     # compute the true w
    mu_y_train = np.zeros(n_class)
    for i in range(n_class):
        mu_y_train[i] = float(len(np.where(train_labels == i)[0]))/m_train
    mu_y_test = np.zeros(n_class)
    for i in range(n_class):
        mu_y_test[i] = float(len(np.where(test_labels == i)[0]))/m_test
    true_w = mu_y_test/mu_y_train
    print('True w is', true_w)
    return true_w

def compute_naive_w(train_labels, n_class, m_train):
    # compute naive label ratio just using the training labels
    mu_y_train = np.zeros(n_class)
    for i in range(n_class):
        mu_y_train[i] = float(len(np.where(train_labels == i)[0]))/m_train
    mu_y_test = 0.1 * np.ones(n_class)
    true_w = mu_y_test/mu_y_train
    print('Naive w is', true_w)
    return true_w

def compute_w_tls(C_yy,mu_y,mu_train_y):
    '''
    TLS
    '''
    n = C_yy.shape[0]
    b = mu_y - mu_train_y
    # augment x with y
    obj = np.zeros((n,n+1))
    obj[0:n, 0:n] = C_yy
    obj[0:n, -1] = b
    # SVD
    u, s, vh = np.linalg.svd(obj, full_matrices=True)
    # calculate theta
    vxx = vh[0:n, -1]
    vyy = vh[-1, -1]
    theta = -vxx/vyy
    
    w = 1 + theta

    w[w<=0] = 0
    print('Estimated w is', w)
    #print(constraints[0].dual_value)
    return w


def acc_perclass(y, predictions, n_class):

    acc = np.zeros(n_class)
    predictions = np.concatenate(predictions)

    for i in range(n_class):
        si = float(len(np.where(y == i)[0]))
        if si != 0:
            acc[i] = float(len(np.where((predictions == i)& (y == i))[0]))/float(len(np.where(y == i)[0]))
        else:
            acc[i] = 0
    return acc

def train_validate_test(args, device, use_cuda, w, train_model, init_state, train_loader, test_loader, validate_loader, test_labels, n_class):
    w = torch.tensor(w)
    train_model.load_state_dict(init_state)
    if use_cuda:
        w = w.cuda().float()
        train_model.cuda()
    else:
        w = w.float()
    
    best_loss = 10
    # model = train_model.to(device)#ConvNet().to(device)
    optimizer = optim.SGD(train_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)
    for epoch in range(1, args.epochs_training + 1):
        train(args, train_model, device, train_loader, optimizer, epoch, weight=w) 
        # save checkpoint
        if epoch > args.epochs_validation:
            # validation
            _, _, loss = test(args, train_model, device, validate_loader, weight=w)
            if loss < best_loss: 
                print('saving model')
                state = {
                    'model': train_model.state_dict(),
                    }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                torch.save(state, './checkpoint/ckpt.pt')
                best_loss = loss
        
    print('\nTesting on test set')
    # read checkpoint
    print('Reading model')
    checkpoint = torch.load('./checkpoint/ckpt.pt')
    train_model.load_state_dict(checkpoint['model'])
    predictions, acc, _ = test(args, train_model, device, test_loader)
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, predictions, average='micro') 
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(test_labels, predictions, average=None) 
    
    acc_per_class = acc_perclass(test_labels, predictions, n_class) 
    print('F1-score (micro): ', f1)
    print('Precision (micro): ', precision)
    print('Recall (micro): ', recall)
    print('Per class accuracy', acc_per_class)
    print('Per class precision', precision_per_class)
    print('Per class recall', recall_per_class)
    print('Per class f-score ', f1_per_class)
    return acc, precision, recall, f1, acc_per_class, precision_per_class, recall_per_class, f1_per_class


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Blackbox Label Shift')
    parser.add_argument('--iterations', type=int, default=20, metavar='N',
                        help='number of iterations to plot comparison plot (default: 20)')
    parser.add_argument('--methods', nargs='+', type = str, help = 'Requred: methods (Choices of rlls, inverse, tls, tw, nw, naive)', required = True)
    parser.add_argument('--data-name', type=str, default='mnist', metavar='N',
                        help='dataset name, mnist or cifar10 (default: mnist)')
    parser.add_argument('--training-size', type=int, default=30000, metavar='N',
                        help='sample size for both training and testing (default: 30000)')
    parser.add_argument('--testing-size', type=int, default=30000, metavar='N',
                        help='sample size for both training and testing (default: 30000)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--shift-type', type = int, default = 2, metavar = 'N',
                        help = 'Label shift type (default: 2)')
    parser.add_argument('--shift-para', nargs='+', type = float,
                        help = 'Required: Label shift paramters (a list)', required=True)
    parser.add_argument('--shift-para-aux', nargs='+', type = float,
                        help = 'Label shift aux paramters (needed for shift-type 7 as target shift param, same size as shift-para)')
    parser.add_argument('--model', type = str, default='MLP', metavar='N',
                        help = 'model type to use for cifar10 (default MLP)')
    parser.add_argument('--epochs-estimation', type=int, default=10, metavar='N',
                        help='number of epochs in weight estimation (default: 10)')
    parser.add_argument('--epochs-training', type=int, default=10, metavar='N',
                        help='number of epochs in training (default: 10)')
    parser.add_argument('--epochs-validation', type=int, default=5, metavar='N',
                        help='number of epochs before run validation set, smaller than epochs training (default: 5)')
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

    num_paras = len(args.shift_para)
    print('Shift-type:', args.shift_type)
    print('# of shift parameters: ', num_paras)
    print('shift parameters: ', args.shift_para)
    if args.shift_type is 7:
        if args.shift_para_aux is None:
            print('Target shift para needed!')
            return
        elif len(args.shift_para_aux) != num_paras:
            print('Num of target shift para wrong!')
            return
        else:
            print('shift parameters aux: ', args.shift_para_aux)
    print('Methods:', args.methods)
    if args.shift_type is not 7 and 'naive' in args.methods:
        print('Naive weights only when shift type is 7.')
        return

    if args.epochs_validation > args.epochs_training:
        print('Epochs of validation should be smaller than epochs of training!')
        return
    # get fixed ho before trying different shift

    if 'inverse' in args.methods or 'rlls' in args.methods or 'tls' in args.methods:
        # training H_0
   
        if args.data_name  == 'mnist':
            raw_data = MNIST_SHIFT('data/mnist', args.training_size, args.testing_size, 1, 0, target_label=9, train=True, download=True,
                transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                               ]))
            D_in = 784
            base_model = Net(D_in, 256, 10)
            
        elif args.data_name == 'cifar10':
            raw_data = CIFAR10_SHIFT('data/cifar10', args.training_size, args.testing_size, 1, 0, target_label=2,
                transform=transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ]), download=True)
            D_in = 3072
            base_model = Net(D_in, 512, 10)
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
        # m_train_t = int(m_train/2)
        print('Training size,', m_train)

        # get labels for future use
        test_labels = raw_data.get_test_label()
        train_labels = raw_data.get_train_label()

   
        # finish data preprocessing
        # estimate weights using training and validation set
        train_loader = data.DataLoader(train_data,
            batch_size=args.batch_size, shuffle=True, **kwargs)
        
        base_model = base_model.to(device)
        optimizer = optim.SGD(base_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)
        print('\nTraining using training_data1 to estimate weights.') 
        for epoch in range(1, args.epochs_estimation + 1):
            train(args, base_model, device, train_loader, optimizer, epoch)

        print("\nfinished learning H_0")

    print('preparing results vectors....')

    if 'rlls' in args.methods:

        acc_w2_vec = torch.zeros([args.iterations, num_paras])
        precision_w2_vec = torch.zeros([args.iterations, num_paras])
        recall_w2_vec = torch.zeros([args.iterations, num_paras])
        f1_w2_vec = torch.zeros([args.iterations, num_paras])
      
        accp_w2_tensor = torch.zeros([args.iterations, num_paras, 10])
        precisionp_w2_tensor = torch.zeros([args.iterations, num_paras, 10])
        recallp_w2_tensor = torch.zeros([args.iterations, num_paras, 10])
        f1p_w2_tensor = torch.zeros([args.iterations, num_paras, 10])

        w2_tensor = torch.zeros([args.iterations, num_paras, 10])

    if 'inverse' in args.methods:

        acc_w1_vec = torch.zeros([args.iterations, num_paras])
        precision_w1_vec = torch.zeros([args.iterations, num_paras])
        recall_w1_vec = torch.zeros([args.iterations, num_paras])
        f1_w1_vec = torch.zeros([args.iterations, num_paras])
      
        accp_w1_tensor = torch.zeros([args.iterations, num_paras, 10])
        precisionp_w1_tensor = torch.zeros([args.iterations, num_paras, 10])
        recallp_w1_tensor = torch.zeros([args.iterations, num_paras, 10])
        f1p_w1_tensor = torch.zeros([args.iterations, num_paras, 10])

        w1_tensor = torch.zeros([args.iterations, num_paras, 10])

    if 'tls' in args.methods:

        acc_w4_vec = torch.zeros([args.iterations, num_paras])
        precision_w4_vec = torch.zeros([args.iterations, num_paras])
        recall_w4_vec = torch.zeros([args.iterations, num_paras])
        f1_w4_vec = torch.zeros([args.iterations, num_paras])
      
        accp_w4_tensor = torch.zeros([args.iterations, num_paras, 10])
        precisionp_w4_tensor = torch.zeros([args.iterations, num_paras, 10])
        recallp_w4_tensor = torch.zeros([args.iterations, num_paras, 10])
        f1p_w4_tensor = torch.zeros([args.iterations, num_paras, 10])

        w4_tensor = torch.zeros([args.iterations, num_paras, 10])

    if 'tw' in args.methods:

        acc_tw_vec = torch.zeros([args.iterations, num_paras])
        precision_tw_vec = torch.zeros([args.iterations, num_paras])
        recall_tw_vec = torch.zeros([args.iterations, num_paras])
        f1_tw_vec = torch.zeros([args.iterations, num_paras])
      
        accp_tw_tensor = torch.zeros([args.iterations, num_paras, 10])
        precisionp_tw_tensor = torch.zeros([args.iterations, num_paras, 10])
        recallp_tw_tensor = torch.zeros([args.iterations, num_paras, 10])
        f1p_tw_tensor = torch.zeros([args.iterations, num_paras, 10])

        tw_tensor = torch.zeros([args.iterations, num_paras, 10])

    if 'nw' in args.methods:

        acc_nw_vec = torch.zeros([args.iterations, num_paras])
        precision_nw_vec = torch.zeros([args.iterations, num_paras])
        recall_nw_vec = torch.zeros([args.iterations, num_paras])
        f1_nw_vec = torch.zeros([args.iterations, num_paras])
      
        accp_nw_tensor = torch.zeros([args.iterations, num_paras, 10])
        precisionp_nw_tensor = torch.zeros([args.iterations, num_paras, 10])
        recallp_nw_tensor = torch.zeros([args.iterations, num_paras, 10])
        f1p_nw_tensor = torch.zeros([args.iterations, num_paras, 10])

    if args.shift_type == 7 and 'naive' in args.methods:

        w3_tensor = torch.zeros([args.iterations, num_paras, 10])

        acc_w3_vec = torch.zeros([args.iterations, num_paras])
        precision_w3_vec = torch.zeros([args.iterations, num_paras])
        recall_w3_vec = torch.zeros([args.iterations, num_paras])
        f1_w3_vec = torch.zeros([args.iterations, num_paras])
      
        accp_w3_tensor = torch.zeros([args.iterations, num_paras, 10])
        precisionp_w3_tensor = torch.zeros([args.iterations, num_paras, 10])
        recallp_w3_tensor = torch.zeros([args.iterations, num_paras, 10])
        f1p_w3_tensor = torch.zeros([args.iterations, num_paras, 10])

        w2_tensor = torch.zeros([args.iterations, num_paras, 10])


    for l in range(num_paras):

        if (args.shift_type == 3) or (args.shift_type == 4):
            alpha = np.ones(10) * args.shift_para[l]
            prob = np.random.dirichlet(alpha)
            shift_para = prob
            shift_para_aux = args.shift_para_aux
        else:
            shift_para = args.shift_para[l]
            shift_para_aux = args.shift_para_aux[l]

        for k in range(args.iterations):
      
            if args.data_name  == 'mnist':
                raw_data = MNIST_SHIFT('data/mnist', args.training_size, args.testing_size, args.shift_type, shift_para, parameter_aux = shift_para_aux, target_label=2, train=True, download=True,
                    transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                                   ]))

                train_model  = Net(D_in, 256, 10)
                train_model = train_model.to(device)
                init_state = copy.deepcopy(train_model.state_dict())
            elif args.data_name == 'cifar10':
                raw_data = CIFAR10_SHIFT('data/cifar10', args.training_size, args.testing_size, args.shift_type, shift_para, parameter_aux = shift_para_aux, target_label=2,
                    transform=transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ]), download=True)

                if args.model == 'Resnet':
                    print('Using Resnet model for predictive tasks')
                    train_model = ResNet18()
                else:
                    train_model = Net(D_in, 256, 10)
                init_state = copy.deepcopy(train_model.state_dict())
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
            # m_train_t = int(m_train/2)
            print('Training size,', m_train)

            # get labels for future use
            test_labels = raw_data.get_test_label()
            train_labels = raw_data.get_train_label()


            # finish data preprocessing

            # compute the true w
            true_w = compute_true_w(train_labels, test_labels, n_class, m_train, m_test)
        

            if 'inverse' in args.methods or 'rlls' in args.methods or 'tls' in args.methods:

                test_loader = data.DataLoader(train_data,
                    batch_size=args.batch_size, shuffle=False, **kwargs)
            
            
                print('\nTesting on training_data2 to estimate C_yy.')
                predictions, acc, _ = test(args, base_model, device, test_loader)

                # compute C_yy 
                C_yy = np.zeros((n_class, n_class)) 
                #print(m_train_v)
                predictions = np.concatenate(predictions)
               
                for i in range(n_class):
                    for j in range(n_class):
                        C_yy[i,j] = float(len(np.where((predictions== i)&(train_labels==j))[0]))/m_train
                    
                mu_y_train_hat = np.zeros(n_class)
                for i in range(n_class):
                    mu_y_train_hat[i] = float(len(np.where(predictions == i)[0]))/m_train

                # print(mu_y_train)
                # print(C_yy)
                # prediction on x_test to estimate mu_y
                print('\nTesting on test data to estimate mu_y.')
                test_loader = data.DataLoader(test_data,
                    batch_size=args.batch_size, shuffle=False, **kwargs)
                predictions, acc, _ = test(args, base_model, device, test_loader)
                mu_y = np.zeros(n_class)
                for i in range(n_class):
                    mu_y[i] = float(len(np.where(predictions == i)[0]))/m_test

                # print(mu_y)

            m_validate = int(0.1*m_train)
            validate_loader = data.DataLoader(data.Subset(train_data, range(m_validate)),
                batch_size=args.batch_size, shuffle=True, **kwargs)
            # 10% validation set
            train_loader = data.DataLoader(data.Subset(train_data, range(m_validate, m_train)),
                batch_size=args.batch_size, shuffle=True, **kwargs)

            if 'rlls' in args.methods:

             

                rho = compute_3deltaC(n_class, m_train, 0.05)
                #alpha = choose_alpha(n_class, C_yy, mu_y, mu_y_train_hat, rho, true_w)
                alpha = 0.001
                w2 = compute_w_opt(C_yy, mu_y, mu_y_train_hat, alpha * rho)
                w2_tensor[k,l,:] = torch.tensor(w2)
                mse2 = np.sum(np.square(true_w - w2))/n_class

                print('MSE (rlls), ', mse2)


                # Learning IW ERM
                print('\nTraining using full training data with estimated weights, testing on test set.')
               
                w = w2
                acc, precision, recall, f1, acc_per, precision_per, recall_per, f1_per = train_validate_test(args, device, use_cuda, w, train_model, init_state, train_loader, test_loader, validate_loader, test_labels, n_class)
                acc_w2_vec[k,l] = acc
                precision_w2_vec[k,l] = precision
                recall_w2_vec[k,l] = recall
                f1_w2_vec[k,l] = f1

                accp_w2_tensor[k,l, :] = torch.tensor(acc_per)
                precisionp_w2_tensor[k,l, :] = torch.tensor(precision_per)
                recallp_w2_tensor[k,l, :] = torch.tensor(recall_per)
                f1p_w2_tensor[k,l, :] = torch.tensor(f1_per)

            if 'inverse' in args.methods:

                w1 = compute_w_inv(C_yy, mu_y)
                w1_tensor[k,l,:] = torch.tensor(w1)

                mse1 = np.sum(np.square(true_w - w1))/n_class 

                print('MSE (inverse), ', mse1)

                if np.abs(mse1 - mse2)  > 0.01:
                    # Compare with using w1
                    w = w1
                    print('\nComparing with using inverse in weight estimation, testing on test set.')
                    acc, precision, recall, f1, acc_per, precision_per, recall_per, f1_per = train_validate_test(args, device, use_cuda, w, train_model, init_state, train_loader, test_loader, validate_loader, test_labels, n_class)
                

                acc_w1_vec[k,l] = acc
                precision_w1_vec[k,l] = precision
                recall_w1_vec[k,l] = recall
                f1_w1_vec[k,l] = f1

                accp_w1_tensor[k,l, :] = torch.tensor(acc_per)
                precisionp_w1_tensor[k,l, :] = torch.tensor(precision_per)
                recallp_w1_tensor[k,l, :] = torch.tensor(recall_per)
                f1p_w1_tensor[k,l, :] = torch.tensor(f1_per)

            if 'tls' in args.methods:
                w4 = compute_w_tls(C_yy, mu_y, mu_y_train_hat)
                print('\nComparing with using total least square, testing on test set.')
                w = w4
                w4_tensor[k,l,:] = torch.tensor(w4)

                mse4 = np.sum(np.square(true_w - w4))/n_class 

                print('MSE (tls), ', mse4)
                acc, precision, recall, f1, acc_per, precision_per, recall_per, f1_per = train_validate_test(args, device, use_cuda, w, train_model, init_state, train_loader, test_loader, validate_loader, test_labels, n_class)
                
                acc_w4_vec[k,l] = acc
                precision_w4_vec[k,l] = precision
                recall_w4_vec[k,l] = recall
                f1_w4_vec[k,l] = f1

                accp_w4_tensor[k,l, :] = torch.tensor(acc_per)
                precisionp_w4_tensor[k,l, :] = torch.tensor(precision_per)
                recallp_w4_tensor[k,l, :] = torch.tensor(recall_per)
                f1p_w4_tensor[k,l, :] = torch.tensor(f1_per)

            if 'nw' in args.methods:

                # Re-train unweighted ERM using full training data, to ensure fair comparison
                print('\nTraining using full training data (unweighted), testing on test set.')
                w = np.ones((10,1))
                acc, precision, recall, f1, acc_per, precision_per, recall_per, f1_per = train_validate_test(args, device, use_cuda, w, train_model, init_state, train_loader, test_loader, validate_loader, test_labels, n_class)
                
                acc_nw_vec[k,l] = acc
                precision_nw_vec[k,l] = precision
                recall_nw_vec[k,l] = recall
                f1_nw_vec[k,l] = f1

                accp_nw_tensor[k,l, :] = torch.tensor(acc_per)
                precisionp_nw_tensor[k,l, :] = torch.tensor(precision_per)
                recallp_nw_tensor[k,l, :] = torch.tensor(recall_per)
                f1p_nw_tensor[k,l, :] = torch.tensor(f1_per)
  
            if 'tw' in args.methods:

                print('\nComparing with using true weight, testing on test set.')
                w = true_w
                tw_tensor[k,l,:] = torch.tensor(true_w)
                acc, precision, recall, f1, acc_per, precision_per, recall_per, f1_per = train_validate_test(args, device, use_cuda, w, train_model, init_state, train_loader, test_loader, validate_loader, test_labels, n_class)
                acc_tw_vec[k,l] = acc
                precision_tw_vec[k,l] = precision
                recall_tw_vec[k,l] = recall
                f1_tw_vec[k,l] = f1

                accp_tw_tensor[k,l, :] = torch.tensor(acc_per)
                precisionp_tw_tensor[k,l, :] = torch.tensor(precision_per)
                recallp_tw_tensor[k,l, :] = torch.tensor(recall_per)
                f1p_tw_tensor[k,l, :] = torch.tensor(f1_per)

            if args.shift_type == 7 and 'naive' in args.methods:
                w3 = compute_naive_w(train_labels, n_class, m_train)
                w = w3
                w3_tensor[k,l,:] = torch.tensor(w3)
                acc, precision, recall, f1, acc_per, precision_per, recall_per, f1_per = train_validate_test(args, device, use_cuda, w, train_model, init_state, train_loader, test_loader, validate_loader, test_labels, n_class)
                acc_w3_vec[k,l] = acc
                precision_w3_vec[k,l] = precision
                recall_w3_vec[k,l] = recall
                f1_w3_vec[k,l] = f1

                accp_w3_tensor[k,l, :] = torch.tensor(acc_per)
                precisionp_w3_tensor[k,l, :] = torch.tensor(precision_per)
                recallp_w3_tensor[k,l, :] = torch.tensor(recall_per)
                f1p_w3_tensor[k,l, :] = torch.tensor(f1_per)
            
    if 'rlls' in args.methods:
        torch.save(acc_w2_vec, 'acc_w2.pt')
        torch.save(precision_w2_vec, 'precision_w2.pt')
        torch.save(recall_w2_vec, 'recall_w2.pt')
        torch.save(f1_w2_vec, 'f1_w2.pt')
        
        torch.save(w2_tensor, 'w2.pt')
        torch.save(accp_w2_tensor, 'w2_accp.pt')
        torch.save(precisionp_w2_tensor, 'w2_precisionp.pt')
        torch.save(recallp_w2_tensor, 'w2_recallp.pt')
        torch.save(f1p_w2_tensor, 'w2_f1p.pt')

    if 'inverse' in args.methods:

        torch.save(acc_w1_vec, 'acc_w1.pt')
        torch.save(precision_w1_vec, 'precision_w1.pt')
        torch.save(recall_w1_vec, 'recall_w1.pt')
        torch.save(f1_w1_vec, 'f1_w1.pt')
        
        torch.save(w1_tensor, 'w1.pt')
        torch.save(accp_w1_tensor, 'w1_accp.pt')
        torch.save(precisionp_w1_tensor, 'w1_precisionp.pt')
        torch.save(recallp_w1_tensor, 'w1_recallp.pt')
        torch.save(f1p_w1_tensor, 'w1_f1p.pt')

    if 'tls' in args.methods:

        torch.save(acc_w4_vec, 'acc_w4.pt')
        torch.save(precision_w4_vec, 'precision_w4.pt')
        torch.save(recall_w4_vec, 'recall_w4.pt')
        torch.save(f1_w4_vec, 'f1_w4.pt')
        
        torch.save(w4_tensor, 'w4.pt')
        torch.save(accp_w4_tensor, 'w4_accp.pt')
        torch.save(precisionp_w4_tensor, 'w4_precisionp.pt')
        torch.save(recallp_w4_tensor, 'w4_recallp.pt')
        torch.save(f1p_w4_tensor, 'w4_f1p.pt')

    if 'tw' in args.methods:

        torch.save(acc_tw_vec, 'acc_tw.pt')
        torch.save(precision_tw_vec, 'precision_tw.pt')
        torch.save(recall_tw_vec, 'recall_tw.pt')
        torch.save(f1_tw_vec, 'f1_tw.pt')
        
        torch.save(tw_tensor, 'tw.pt')
        torch.save(accp_tw_tensor, 'tw_accp.pt')
        torch.save(precisionp_tw_tensor, 'tw_precisionp.pt')
        torch.save(recallp_tw_tensor, 'tw_recallp.pt')
        torch.save(f1p_tw_tensor, 'tw_f1p.pt')

    if 'nw' in args.methods:

        torch.save(acc_nw_vec, 'acc_nw.pt')
        torch.save(precision_nw_vec, 'precision_nw.pt')
        torch.save(recall_nw_vec, 'recall_nw.pt')
        torch.save(f1_nw_vec, 'f1_nw.pt')
      
        torch.save(accp_nw_tensor, 'nw_accp.pt')
        torch.save(precisionp_nw_tensor, 'nw_precisionp.pt')
        torch.save(recallp_nw_tensor, 'nw_recallp.pt')
        torch.save(f1p_nw_tensor, 'nw_f1p.pt')

    if args.shift_type == 7 and 'naive' in args.methods:
        torch.save(acc_w3_vec, 'acc_w3.pt')
        torch.save(precision_w3_vec, 'precision_w3.pt')
        torch.save(recall_w3_vec, 'recall_w3.pt')
        torch.save(f1_w3_vec, 'f1_w3.pt')
        
        torch.save(w3_tensor, 'w3.pt')
        torch.save(accp_w3_tensor, 'w3_accp.pt')
        torch.save(precisionp_w3_tensor, 'w3_precisionp.pt')
        torch.save(recallp_w3_tensor, 'w3_recallp.pt')
        torch.save(f1p_w3_tensor, 'w3_f1p.pt')







if __name__ == '__main__':
    main()



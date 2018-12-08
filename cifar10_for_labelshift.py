from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torch
import torch.utils.data as data
from torchvision import datasets


class CIFAR10_SHIFT(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    @property
    def targets(self): 
        return self.labels
       

    def __init__(self, root, training_size, testing_size, shift_type, parameter, parameter_aux=None, target_label=None,
                 transform=None, target_transform=None,
                 download=False):
        """
        Converted function from original torch cifar10 class :
        new parameters:
        sample_size: int, sample size of both training and testing set
        shift_type: int, 1 for knock one shift, 2 for tweak one shift and 3 for dirichlet shift
                         4 for dirichlet shift on the testing set, training is uniform
        parameter: float in [0, 1], delta for knock one shift, delete target_label by delta
                                    or, rho for tweak one shift, set target_label probability as rho, others even
                                    or, alpha for dirichlet shift
        target_label: int, target label for knock one and tweak one shift

        REMOVED paramter: train -- since we will merge training and testing 
        """
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # now load the picked numpy arrays
        # merge the training and testing together 
        
        raw_data = []
        raw_labels = []
        for fentry in self.train_list:
            f = fentry[0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            raw_data.append(entry['data'])
            if 'labels' in entry:
                raw_labels += entry['labels']
            else:
                raw_labels += entry['fine_labels']
                fo.close()
           
        
        f = self.test_list[0][0]
        file = os.path.join(self.root, self.base_folder, f)
        fo = open(file, 'rb')
        if sys.version_info[0] == 2:
            entry = pickle.load(fo)
        else:
            entry = pickle.load(fo, encoding='latin1')
        raw_data.append(entry['data'])
        if 'labels' in entry:
            raw_labels += entry['labels']
        else:
            raw_labels += entry['fine_labels']
        fo.close()

        self._load_meta()
        # merge training and testing
        raw_data = np.concatenate(raw_data)
        raw_labels = np.asarray(raw_labels)
        # # creat label shift
        indices = np.random.permutation(60000)


        if training_size > 30000 or testing_size > 30000:
            raise RuntimeError("Supported setting is sample size smaller than 30000.")
         
        if shift_type==1 or shift_type ==2 or shift_type ==3 or shift_type ==6:

            m_test = testing_size

            test_indices = indices[0 : m_test]
            train_indices = indices[m_test :]
        elif shift_type !=7:
            m_train = training_size

            train_indices = indices[0 : m_train]
            test_indices = indices[m_train :]
        else:
            m_train = 30000
            m_test = 30000
            train_indices = indices[0 : m_train]
            test_indices = indices[m_train :]


        test_data = raw_data[(test_indices,)]
        test_labels = raw_labels[(test_indices,)]

        train_data = raw_data[(train_indices,)]
        train_labels = raw_labels[(train_indices,)]
        num_train = len(train_data)

        if shift_type == 1:
            if target_label == None:
                raise RuntimeError("There should be a target label for the knock one shift.")
            indices_target = np.where(train_labels == target_label)[0]
            num_target = len(indices_target)
            num_knock = int(num_target * parameter)    
            train_data = np.delete(train_data, indices_target[0:num_knock], 0)
            train_labels = np.delete(train_labels, indices_target[0:num_knock])
            m_train = len(train_labels)
            
        elif shift_type == 2:
            if target_label == None:
                raise RuntimeError("There should be a target label for the tweak one shift.")
            # use the number of target label to decide the total number of the training samples
            
            if parameter < (1.0-parameter)/9 :
                target_label = (target_label + 1)%10
            indices_target = np.where(train_labels == target_label)[0]
            num_target = len(indices_target)    
            num_train = int(num_target/parameter)
            #################
            # num_train = sample_size
            # num_target = int(num_train * parameter)
 
            num_remain = num_train - num_target
            # even on other labels
            num_i = int(num_remain/9)
            indices_train = np.empty((0,1), dtype = int)

            for i in range(10):
                indices_i = np.where(train_labels == i)[0]
                if i != target_label:
                    indices_i = indices_i[0:num_i] 
                else:
                    indices_i = indices_i[0:num_target] 
                indices_train = np.append(indices_train, indices_i)
            
            shuffle = np.random.permutation(len(indices_train))
            train_data = train_data[(indices_train[shuffle],)]
            train_labels = train_labels[(indices_train[shuffle],)]
            m_train = len(train_labels)

        elif shift_type == 3:
            # use the maximum prob to decide the total number of training samples
            target_label = np.argmax(parameter)
            print(parameter)

            indices_target = np.where(train_labels == target_label)[0]
            num_target = len(indices_target)
            prob_max = np.amax(parameter)    
            num_train = int(num_target/prob_max)
            indices_train = np.empty((0,1), dtype = int)
            for i in range(10):
                num_i = int(num_train * parameter[i])
                indices_i = np.where(train_labels == i)[0]
                indices_i = indices_i[0:num_i] 
                indices_train = np.append(indices_train, indices_i)

            shuffle = np.random.permutation(len(indices_train))
            train_data = train_data[(indices_train[shuffle],)]
            train_labels = train_labels[(indices_train[shuffle],)]
            m_train = len(train_labels)

        elif shift_type == 4:

            # use the maximum prob to decide the total number of training samples
            target_label = np.argmax(parameter)
            print('Dirichlet shift with prob,', parameter)

            indices_target = np.where(test_labels == target_label)[0]
            num_target = len(indices_target)
            prob_max = np.amax(parameter)    
            m_test = int(num_target/prob_max)
            indices_test = np.empty((0,1), dtype = int)

            for i in range(10):
                num_i = int(m_test * parameter[i])
                indices_i = np.where(test_labels == i)[0]
                indices_i = indices_i[0:num_i] 
                indices_test = np.append(indices_test, indices_i)   
            shuffle = np.random.permutation(len(indices_test))
            test_data = test_data[(indices_test[shuffle],)]
            test_labels = test_labels[(indices_test[shuffle],)]
            m_test = len(test_labels)

        elif shift_type == 5:
            if target_label == None:
                raise RuntimeError("There should be a target label for the tweak one shift.")
            # use the number of target label to decide the total number of the training samples
            
            if parameter < (1.0-parameter)/9 :
                target_label = (target_label + 1)%10
            indices_target = np.where(test_labels == target_label)[0]
            num_target = len(indices_target)    
            num_test = int(num_target/parameter)

            num_remain = num_test - num_target
            # even on other labels
            num_i = int(num_remain/9)
            indices_test = np.empty((0,1), dtype = int)

            for i in range(10):
                indices_i = np.where(test_labels == i)[0]
                if i != target_label:
                    indices_i = indices_i[0:num_i] 
                else:
                    indices_i = indices_i[0:num_target] 
                indices_test = np.append(indices_test, indices_i)
            
            shuffle = np.random.permutation(len(indices_test))
            test_data = test_data[(indices_test[shuffle],)]
            test_labels = test_labels[(indices_test[shuffle],)]
            m_test = len(test_labels)

        elif shift_type == 6:
            # randomly choose 2 target labels
            target_label = np.linspace(1, parameter, parameter) 
            para = 0.01
            prob = (1 - len(target_label) *para)/(10 - len(target_label))
            indices_target = np.where(train_labels == target_label[0])[0]
            num_target = len(indices_target)
            num_train = int(num_target/prob)
            num_target = int(num_train*prob)

            # even on other labels
            num_i = int(num_train * para)
            indices_train = np.empty((0,1), dtype = int)

            for i in range(10):
                indices_i = np.where(train_labels == i)[0]
                if i in target_label:
                    indices_i = indices_i[0:num_i] 
                else:
                    indices_i = indices_i[0:num_target] 
                indices_train = np.append(indices_train, indices_i)
            
            shuffle = np.random.permutation(len(indices_train))
            train_data = train_data[(indices_train[shuffle],)]
            train_labels = train_labels[(indices_train[shuffle],)]
            m_train = len(train_labels)
        elif shift_type == 7:
           
            if parameter < (1.0-parameter)/9 :
                target_label = (target_label + 1)%10
            indices_target = np.where(train_labels == target_label)[0]
            num_target = len(indices_target)    
            num_train = int(num_target/parameter)
            #################
            # num_train = sample_size
            # num_target = int(num_train * parameter)
 
            num_remain = num_train - num_target
            # even on other labels
            num_i = int(num_remain/9)
            indices_train = np.empty((0,1), dtype = int)

            for i in range(10):
                indices_i = np.where(train_labels == i)[0]
                if i != target_label:
                    indices_i = indices_i[0:num_i] 
                else:
                    indices_i = indices_i[0:num_target] 
                indices_train = np.append(indices_train, indices_i)
            
            shuffle = np.random.permutation(len(indices_train))
            train_data = train_data[(indices_train[shuffle],)]
            train_labels = train_labels[(indices_train[shuffle],)]
            m_train = len(train_labels)
            print(m_train)
            # testint portion

            if parameter_aux < (1.0-parameter_aux)/9 :
                target_label = (target_label + 1)%10
            indices_target = np.where(test_labels == target_label)[0]
            num_target = len(indices_target)    
            num_test = int(num_target/parameter_aux)
            #################
            # num_train = sample_size
            # num_target = int(num_train * parameter)
 
            num_remain = num_test - num_target
            # even on other labels
            num_i = int(num_remain/9)
            indices_test = np.empty((0,1), dtype = int)
            print(num_i)
            for i in range(10):
                indices_i = np.where(test_labels == i)[0]
                if i != target_label:
                    indices_i = indices_i[0:num_i] 
                else:
                    indices_i = indices_i[0:num_target] 
                indices_test = np.append(indices_test, indices_i)
            
            shuffle = np.random.permutation(len(indices_test))
            test_data = test_data[(indices_test[shuffle],)]
            test_labels = test_labels[(indices_test[shuffle],)]
            m_test = len(test_labels)
            print(m_test)

        else:
            raise RuntimeError("Invalid shift type.")

        if training_size > m_train:
            training_size = m_train
        if testing_size > m_test:
            testing_size = m_test
        train_data = train_data[range(training_size)]
        train_labels = train_labels[range(training_size)]
        
        test_data = test_data[range(testing_size)]
        test_labels = test_labels[range(testing_size)]

        features = np.concatenate((test_data, train_data))
        features = features.reshape((-1, 3, 32, 32))
        features = features.transpose((0, 2, 3, 1))  # convert to HWC
        self.data = features
        self.labels = np.concatenate((test_labels, train_labels))
        self.train_labels = train_labels
        self.test_labels = test_labels

        self.m_test = testing_size
        self.m_train = training_size


    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not datasets.utils.check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        
        img, target = self.data[index], self.labels[index]
        

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
      
        return len(self.data)

    def get_train_label(self):
        return self.train_labels
    def get_test_label(self):
        return self.test_labels

    def get_testsize(self):
        return self.m_test
    

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not datasets.utils.check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.root
        datasets.utils.download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

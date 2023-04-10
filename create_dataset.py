import torchvision.transforms as transforms
import torch.utils.data as data
import copy
from torchvision.transforms.autoaugment import AutoAugmentPolicy
from torchvision import datasets
from torch.utils.data import random_split



def get_data_loaders(batch_size, root = '.data', valid_ratio = 0.1):
  
    #means and stds for the 3 channels of the dataset
    means = (0.4914, 0.4822, 0.4465)
    stds = (0.2023, 0.1994, 0.2010)

    #   Performs the following data augmentations on the training set:
    #       1.  Random Rotation:  
    #       2.  RandomHorizontalFlip
    #       3.  Random Crop
    #       4.  AutoAugment: AutoAugmention policies learned from the CIFAR10 dataset
    #           AutoAugment: Learning Augmentation Strategies from Data" <https://arxiv.org/pdf/1805.09501.pdf>`
    train_transforms = transforms.Compose([
                           transforms.RandomRotation(5),
                           transforms.RandomHorizontalFlip(0.5),
                           transforms.RandomCrop(32, padding = 2),
                           transforms.AutoAugment(AutoAugmentPolicy.CIFAR10),
                           transforms.ToTensor(),  # convert data to torch.FloatTensor: CxHxW
                           transforms.Normalize(mean = means, std = stds, inplace=True), #Range is in [-1,1], data = (data - mean)/std
                       ])

    test_transforms = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(mean = means, std = stds, inplace=True),
                       ])

    # choose the training and test datasets
    train_data = datasets.CIFAR10(root, 
                              train = True, 
                              download = True, 
                              transform = train_transforms)
    
    test_data = datasets.CIFAR10(root, 
                             train = False, 
                             download = True, 
                             transform = test_transforms)
    
    # obtain number of training/validation examples that will be used to create the respective datasets 
    n_train_examples = int((1 - valid_ratio) * len(train_data))
    n_valid_examples = int(valid_ratio* len(train_data))

    train_data, valid_data = random_split(train_data, [n_train_examples, n_valid_examples])
    
    #Creates a deep copy
    valid_data = copy.deepcopy(valid_data)  
    valid_data.dataset.transform = test_transforms

    # load training data in batches
    train_dataloader = data.DataLoader(train_data,
                                 shuffle = True,
                                 batch_size = batch_size,
                                 num_workers=2, 
                                 pin_memory=True)   #pin_memory speeds transfer of data from CPU to GPU

    # load validation data in batches
    valid_dataloader = data.DataLoader(valid_data,
                                 batch_size = batch_size,
                                 num_workers=2, 
                                 pin_memory=True)

    # load test data in batches
    test_dataloader = data.DataLoader(test_data,
                                batch_size = batch_size,
                                num_workers=2, 
                                pin_memory=True)
  
    return train_dataloader, valid_dataloader, test_dataloader
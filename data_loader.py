import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.autoaugment import AutoAugmentPolicy

ROOT = './data'
BATCH_SIZE = 100
VALID_RATIO = 0.9

def load_valid_train(data_dir=ROOT):
    #means and stds for the 3 channels of the dataset
    means = (0.4914, 0.4822, 0.4465)
    stds = (0.2023, 0.1994, 0.2010)

    train_transforms = transforms.Compose([
                           transforms.AutoAugment(AutoAugmentPolicy.CIFAR10),
                           transforms.ToTensor(),  #CxHxW
                           transforms.Normalize(mean = means, std = stds, inplace=True), #Range is in [-1,1], data = (data - mean)/std
                       ])
    
    test_transforms = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(mean = means, std = stds, inplace=True),
                       ])
    
    trainset = torchvision.datasets.CIFAR10(root=ROOT, 
                                            train=True, 
                                            download=True, 
                                            transform=train_transforms)
    trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=ROOT, 
                                           train=False, 
                                           download=True, 
                                           transform=test_transforms)
    testloader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


    return trainloader, testloader
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.autoaugment import AutoAugmentPolicy
import torch.nn as nn
import torch.optim as optim
import copy
from utils import*
from Model import*

EPOCHS = 5
BATCH_SIZE = 100
VALID_RATIO = 0.9
LEARNING_RATE = 1e-3

def load_valid_train(ROOT = './data'):
    #means and stds for the 3 channels of the dataset
    means = (0.4914, 0.4822, 0.4465)
    stds = (0.2023, 0.1994, 0.2010)

    train_transforms = transforms.Compose([
                           transforms.AutoAugment(AutoAugmentPolicy.CIFAR10),
                           transforms.ToTensor(),  #CxHxW
                           transforms.Normalize(mean = means, std = stds, inplace=True), #Range is in [-1,1], data = (data - mean)/std
                       ])
    
    val_transforms = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(mean = means, std = stds, inplace=True),
                       ])
    
    train_set = torchvision.datasets.CIFAR10(root=ROOT, 
                                            train=True, 
                                            download=True, 
                                            transform=train_transforms)

    n_train_examples = int(len(train_set) * VALID_RATIO)
    n_valid_examples = len(train_set) - n_train_examples

    train_data, valid_data = torch.utils.data.random_split(train_set, 
                                           [n_train_examples, n_valid_examples],
                                           generator=torch.Generator().manual_seed(123456))

    valid_data = copy.deepcopy(valid_data) #deepcopy 
    valid_data.dataset.transform = val_transforms  

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    valid_loader = torch.utils.data.DataLoader(
        valid_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


    return train_loader, valid_loader

def train(epoch, model, train_loader, optimizer, criterion, device):
    train_loss = 0.0
    train_acc = 0
    steps = 0
    
    model.train()
    
    for i,(images, labels) in enumerate (train_loader):
        # get the inputs; data is a list of [inputs, labels]
        images = images.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize       
        predictions = model(images)
        loss = criterion(predictions, labels)
        acc = calculate_accuracy(predictions, labels)
        loss.backward()
        optimizer.step()
        
        #statistics
        train_loss += loss.item()
        train_acc += acc.item()
        steps += 1

         
        progress_bar(i, len(train_loader), 'Loss: %.3f | Acc: %.3f'
                     % (train_loss/(i+1), 100.*train_acc))
        
    return train_loss / steps, train_acc / steps

def validation(model, valid_loader, criterion, device):
    
    valid_loss = 0.0
    valid_acc = 0
    steps = 0
    
    model.eval()
    
    with torch.no_grad():
        
        for i,(images, labels) in enumerate (valid_loader):

            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images)

            loss = criterion(predictions, labels)

            acc = calculate_accuracy(predictions, labels)

            valid_loss += loss.item()
            valid_acc += acc.item()
            steps += 1
        
    return valid_loss / steps, valid_acc / steps

def main():
    # Geting cpu or gpu device for training:
    model = ResNet18()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_loader, valid_loader = load_valid_train()

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    for epoch in range(EPOCHS):
        train_loss, train_acc = train(epoch, model, train_loader, optimizer, criterion, device)
        valid_loss, valid_acc = validation(model, valid_loader, criterion, device)
        print(f"Epoch: {epoch} \n Train_Loss: {train_loss} \t Train_Acc: {train_acc} \n Valid_Loss: {valid_loss} \t Valid_acc: {valid_acc}") 

if __name__ == '__main__':
    main()
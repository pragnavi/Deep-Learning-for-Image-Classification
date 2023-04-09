import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from torchsummary import summary
import time
from Resnet import*
from create_dataset import*
from utils import*

BATCH_SIZE = 64
EPOCHS = 2
INPUT_SIZE = (3,32,32)
LEARNING_RATE = 1e-3


def train(model, iterator, optimizer, criterion, device):
    
    # Tracks training loss and accuracy as model trains
    epoch_loss = 0
    epoch_acc = 0
    #lrs = []

    # Prep model for training
    model.train()

    for i, (x, y) in enumerate(iterator):
        print(str(i) + ' ' + str(len(iterator)))
        # Move inputs and labels to device
        x = x.to(device)
        y = y.to(device)
        
        # Clear the gradients of all optimized variables
        optimizer.zero_grad()

        # Forward pass: compute predicted outputs by passing inputs to the model        
        y_pred = model(x)
        
        # Calculate the loss
        loss = criterion(y_pred, y)
        
        # Calculate the accuracy
        acc = calculate_accuracy(y_pred, y)
        
        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()

        #OneCycleLr Scheduler
        #scheduler.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion, device):
    # Keeps track of validation loss and  accuracy
    epoch_loss = 0
    epoch_acc = 0
    
    # Prep model for evaluation
    model.eval()
    
    with torch.no_grad():
        
        for (x, y) in iterator:

            x = x.to(device)
            y = y.to(device)

            # Forward Pass
            y_pred = model(x)
            # Calculate the validation loss
            loss = criterion(y_pred, y)
            #Calculate the validation accuracy
            acc = calculate_accuracy(y_pred, y)

            #Record validation loss
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def main():

    # Sets seeds for functions with random components
    seed_everything()

    # Specify device: check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load and batch the data
    train_dataloader, valid_dataloader, _ = get_data_loaders(BATCH_SIZE)

    # Specify the model, loss function, and optimizer; move model to device
    model = ResNet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Specify scheduler
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    # Check total number of parameters
    summary(model, input_size=INPUT_SIZE)

    # Train and Validate the model:

    # Tracks average training loss and accuracy per epoch
    #  and tracks average validation loss and accuracy per epoch
    Train_Loss = []
    Train_Acc = []
    Val_Loss = []
    Val_Acc = []


    # Tracks best validation loss
    best_valid_loss = float('inf')
    for epoch in range(EPOCHS):
        # Epoch start time
        start_time = time.monotonic()


        train_loss, train_acc = train(model, train_dataloader, optimizer, criterion, device)
        valid_loss, valid_acc = evaluate(model, valid_dataloader, criterion, device)
  

        # OnReducePlateau is a scheduler monitoring validation loss
        scheduler.step(valid_loss)

        # Saves model with best validation loss in a state dictionary
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(),'model.pt')
        # Epoch end time
        end_time = time.monotonic()

        # Prints training/validation statistics as well as epoch time
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch: {epoch+1:02} || EpochTime: {epoch_mins}m{epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} || Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val Loss: {valid_loss:.3f} ||  Val Acc: {valid_acc*100:.2f}%')

        Train_Loss.append(train_loss)
        Train_Acc.append(train_acc)
        Val_Loss.append(valid_loss)
        Val_Acc.append(valid_acc)


    # Prints training loss and validation loss 
    get_loss_plot(Train_Loss, Val_Loss)

if __name__ == "__main__":
    main()
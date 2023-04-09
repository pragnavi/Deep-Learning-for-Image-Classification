import torch
from train import evaluate
from create_dataset import get_data_loaders
from Resnet import*

BATCH_SIZE = 64

def main():  
    # Specify device: check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load and batch test data
    _, _, test_dataloader = get_data_loaders(BATCH_SIZE)

    # Specify model and loss function
    model = ResNet18().to(device)
    criterion = nn.CrossEntropyLoss()

    # Load saved model's parameters
    model.load_state_dict(torch.load('model.pt'))

    # Calculate test loss and accuracy
    test_loss, test_acc = evaluate(model, test_dataloader, criterion, device)


    print(f'Test Loss: {test_loss:.3f} || Test Acc: {test_acc*100:.2f}%')

if __name__ == '__main__':  
    main()

import matplotlib.pyplot as plt

# accuracy is the fraction of predictions the model got correct
def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

# time interval for each epoch
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# visualize the loss as the network trained
def get_loss_plot(train_loss, val_loss):
    plt.plot(range(len(train_loss),train_loss))
    plt.plot(range(len(train_loss),val_loss))
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title('Loss')
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.show()
    plt.savefig('loss_plot.png')
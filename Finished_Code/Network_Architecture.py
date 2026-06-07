import torch
import torch.nn as nn
import torch.nn.functional as F
from Finished_Code.logger import *

def createMYLogger():
    outlogger = logging.getLogger(__name__)
    outlogger.setLevel(logging.DEBUG) # Logger will process messages from DEBUG up

    file_handler = logging.FileHandler("log.log")
    file_handler.setLevel(logging.DEBUG) # Handler will write messages from DEBUG up

    outlogger.addHandler(file_handler)
    outlogger.debug("test")
    return outlogger

logger = createMYLogger()

#make Network more maluable 
class Net(nn.Module):
    """ 
    2D Convolutional Neural Network with 2 convolutional layers, 2 fully connected layers, 2 max pooling layers, and square kernels. 
    This network uses a mix of relu and softmax activation functions. 

    Attributes:
      kernel_size_conv: An integer to set the dimesions of the square kernel for the convolutional layers
      kernel_size_pool: An integer to set the dimesions of the square kernel for the pooling layers
      reshape_dims: An integer that sets the in_features of the first fully connected layer
      fc_dims: An integer that sets the out_features of the first fully connected layer
      conv1: creates the first 2D convolutional layer
      conv2: creates the second 2D convolutional layer
      fc1: creates the first fully connected layer
      fc2: creates the second fully connected layer
    """
    def __init__(self, in_channel, out_channel1, out_channel2, final_out, kernel_size_conv, kernel_size_pool, reshape_dims, fc_dims):
        """Inits Net

        Args:
          in_channel: the number of channels going into the first convolutional layer (should be <name_of_dataset>.shape[1])
          out_channel1: the number of channels going out of the first convolutional layer (a large increase- ex. 3 to 50- is reccommended)
          out_channel2: the number of channels going out of the second convolutional layer
          final_out: the out_features from the second fully connected layer
        """
        super(Net, self).__init__()
        self.kernel_size_conv= kernel_size_conv
        self.kernel_size_pool = kernel_size_pool
        self.reshape_dims = reshape_dims
        self.fc_dims = fc_dims
        self.conv1 = nn.Conv2d(in_channel, out_channel1, kernel_size_conv)
        self.conv2 = nn.Conv2d(out_channel1, out_channel2, kernel_size_conv)
        self.fc1 = nn.Linear(reshape_dims, fc_dims)
        self.fc2 = nn.Linear(fc_dims, final_out)
    def forward(self, x):
        """ Runs a batch of data through the neural network

        Args:
          x: batch of dataset

        Returns:
          a tensor that represents the output and gradient of x after one forward action is taken on it
        """

        #try mean filter before conv layers to downsample (k size 4)
        #x= F.interpolate(x, scale_factor=0.25, mode = 'bilinear', antialias= True)
        x = F.relu(F.max_pool2d(self.conv1(x), self.kernel_size_pool))
        x = F.relu(F.max_pool2d(self.conv2(x), self.kernel_size_pool))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=None) 
    





def train(epoch, network, train_loader, optimizer, train_losses, train_counter, log_interval=10, prints=True):
  """Runs the train set of the data with the option to print train stats and loss info
  Args:
    epoch: integer to determine number of train and test cycles needed
    network: Net object that will be the network to train on
    train_loader: DataLoader object that includes the train data
    optimizer: Optimizer object that will act as the network's optimizer
    train_losses: a list used to keep track of the train loss data
    train_counter: a list used to keep track of the training statistics
    log_interval: an integer to determine how many iterations of training must past before printing stats
    prints: a boolean to determine if the train statstics should be printed

  Returns:
    Tuple with:
      train_losses: a list used to keep track of the train loss data
      train_counter: a list used to keep track of the training statistics
  """
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader): 
    optimizer.zero_grad()
    output = network(data)
    loss = F.nll_loss(output, target.long()) #negative log likelihood loss
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      if(prints):
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch, batch_idx * len(data), len(train_loader.dataset),
          100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*data.shape[0]) + ((epoch-1)*len(train_loader.dataset))) #Check what 64 is!!!!!!!!!!!!!!!!!!!!!!!!!!
      #torch.save(network.state_dict(), '/results/model.pth')
      #torch.save(optimizer.state_dict(), '/results/optimizer.pth')
  return train_losses, train_counter
      
def test(network, test_loader, test_losses, prints=True):
  """Runs the test set of the data with the option to print accuracy and loss info
  Args:
    network: Net object that will be the network to test on
    test_loader: DataLoader object that includes the test data
    test_losses: a list used to keep track of the test loss data
    prints: a boolean to determine if the train statstics should be printed

  Returns:
    Tuple with:
      test_losses: a list used to keep track of the test loss data
      y_test: a list of the ground truth categorizations
      predictions: a list of predicted categorizations made by the neural network
  """
  y_test = []
  predictions = [] 
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      y_test.append(target)
      output = network(data)
      test_loss += F.nll_loss(output, target.long(), size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      logger.debug(f"output data {output.data}")
      logger.debug(f"output data max {output.data.max(1, keepdim=True)}")
      logger.debug(f"pred {pred}")
      predictions.append(pred)
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  if(prints):
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
      test_loss, correct, len(test_loader.dataset),
      100. * correct / len(test_loader.dataset)))
  return test_losses, y_test, predictions




import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns
from Finished_Code.Network_Architecture import *


def computeNextDimsCNN(Hin, Win, kernel_size_conv, padding:tuple, dilation:tuple, kernel_size_pool, numLayers):
    """Computes the dimensions of an image after being put through <numLayers> amount of convolutional layers
    Args:
        Hin: an integer representing height in
        Win: an integer representing weight in
        kernel_size_conv: an integer representing the height of the square convolutional kernel
        padding: an integer representing the amount of padding used in the layer(s)
        dilation: an integer representing the amount of dilation used in the layer(s)
        kernel_size_pool: an integer representing the height of the square pooling kernel
        numLayers: an integer representing the number of layers this calculation should be repeated for

    Returns:
        ouput: the adjusted dimesions for the height of the square image 
    """
    Hout = Hin
    Wout = Win
    for i in range(numLayers):
        
        Hout = int(abs((Hout + 2 * padding[0] - dilation[0] * (kernel_size_conv-1)-1)))
        Hout = int(abs(Hout - (kernel_size_pool -1))/kernel_size_pool) +1
        Wout = int(abs((Wout + 2 * padding[1] - dilation[1] * (kernel_size_conv-1)-1)))
        Wout = int(abs(Wout - (kernel_size_pool -1))/kernel_size_pool) +1


        logger.debug(f"Hout {Hout}")
        
        logger.debug(f"Wout {Wout}")


    return Hout, Wout



def defDims(Hin, Win, kernel_size_conv, batch_size_test, out_channel2, padding = (0,0), dilation = (1,1), kernel_size_pool = 1, numLayers = 1):
    """Computes the dimensions required for the fully connected layers to avoid matrix multiplication issues
    Args:
        Hin: an integer representing height in
        Win: an integer representing weight in
        kernel_size_conv: an integer representing the height of the square convolutional kernel
        batch_size_test: an integer to determine what the size of each test batch should be
        out_channel2: the number of channels going out of the second convolutional layer
        padding: an integer representing the amount of padding used in the layer(s)
        dilation: an integer representing the amount of dilation used in the layer(s)
        kernel_size_pool: an integer representing the height of the square pooling kernel
        numLayers: an integer representing the number of layers this calculation should be repeated for

    Returns:
        Tuple with:
            reshape_dims: An integer that sets the in_features of the first fully connected layer
            fc_dims: An integer that sets the out_features of the first fully connected layer
    """
    h_out = Hin
    w_out = Win
    h_out, w_out = computeNextDimsCNN(Hin, Win, kernel_size_conv, padding = padding, dilation = dilation, kernel_size_pool=kernel_size_pool, numLayers=numLayers)
    final_dim = h_out* w_out * out_channel2
    fc_dims = batch_size_test//out_channel2
    return fc_dims, final_dim

def createSNRDict(maxSnr, minSnr, ranges):
    """Creates a dictionary that represents categories of SNRs whose size is determined by ranges 
    and amount of categories is determined by maxSnr and minSnr. The dictionary can be accessed using index 
    or the minimum of the SNR range. 

    ex: the third item in a dictionary named "SNR 2.5-5" can be accessed with the key 2 or the key 2.5
    
    Args: 
        maxSnr: an integer or float to set the maximum signal to noise ratio included in the dictionary
        minSnr: an integer or float to set the minimum signal to noise ratio included in the dictionary
        ranges: an integer or float to set the amount of signal to noise in each category of the dictionary

    Returns: 
        Tuple with:
            snrDict: dictionary with signal to noise categories (use createSNRDict)
            index: an integer representing the number of unique items in the dictionary

    Raises: 
        AssertionError: occurs when maxSnr <= minSnr OR ranges is <= 0
    """
    assert maxSnr > minSnr
    assert ranges > 0
    snrDict = {}
    counter = minSnr
    index = 0
    while (counter+ranges <= maxSnr):

        snrDict[counter] = f"SNR {counter}-{counter+ranges}"
        snrDict[index] = f"SNR {counter}-{counter+ranges}"
        index+= 1
        
        counter += ranges
        



    if ((maxSnr - minSnr)/ranges != 0):
        snrDict[counter] = f"SNR {counter}-{maxSnr}"
        snrDict[index] = f"SNR {counter}-{maxSnr}"
        index+=1
    
    return snrDict, index



def plotCMHeatmap(cm, title, states):
    """plots a Confusion Matrix that includes accuracy, recall, and precision

    Args: 
        cm: list object in confusion matrix format as used by PyTorch
        snrDict: dictionary with signal to noise categories (use createSNRDict)
        dictKey: the key for which item in the dictionary should be used
        states: the classification states
    """
    total = np.sum(cm)
    labels = [[f"{val:0.0f}\n{val / total:0.0%}" for val in row] for row in cm]
     
    
    ax = sns.heatmap(cm, annot=labels, cmap='Reds', fmt = "",
                     xticklabels=states, yticklabels=states, cbar=False)
    plt.title(title)
    ax.tick_params(labeltop=True, labelbottom=False, length=0)

    
    # matrix for the extra column and row
    f_mat = np.zeros((cm.shape[0] + 1, cm.shape[1] + 1))
    f_mat[:-1, -1] = np.diag(cm) / np.sum(cm, axis=1)  
    f_mat[-1, :-1] = np.diag(cm) / np.sum(cm, axis=0)  
    f_mat[-1, -1] = np.trace(cm) / np.sum(cm)  
    
    f_mask = np.ones_like(f_mat)  
    f_mask[:, -1] = 0  
    f_mask[-1, :] = 0  
    
    # matrix for coloring the heatmap
    # only last row and column will be used due to masking
    f_color = np.ones_like(f_mat)
    f_color[-1, -1] = 0  # lower right gets different color
    
    # matrix of annotations, only last row and column will be used
    f_annot = [[f"{val:0.0%}" for val in row] for row in f_mat]
    f_annot[-1][-1] = "Acc.:\n" + f_annot[-1][-1]
    
    sns.heatmap(f_color, mask=f_mask, annot=f_annot, fmt = "", 
                xticklabels=states + ["Recall"],
                yticklabels=states + ["Precision"],
                ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('Actual Label')
    plt.tight_layout()
    plt.show()


        
        

    







def makeAllCMPlots(ranges, y_test, predictions, labels, test_indices, snr, states):
    """Creates the Confusion Matrix plots for each signal to noise range.

    Args:
        ranges: an integer or float to set the amount of signal to noise in each category of the dictionary
        y_test: a list of the ground truth categorizations**
        predictions: a list of predicted categorizations made by the neural network**
        **the index of these two lists match (eg: index 0 in both lists refers to the same tensor in the data)
        snrMax: an integer or float to set the maximum signal to noise ratio included in the dictionary
        snrMin: an integer or float to set the minimum signal to noise ratio included in the dictionary
        labels: a np array of all the ground truth categorizations
        test_indices: a np array of all the indices that correspond to the test DataLoader (can be used to reference labels and snr)
        snr: a np array of all the signal to noise ratio values 
        states: the classification states

    Raises:
        AssertionError: occurs when indexing is innaccurate accross arrays
    """
    y_test_arr = np.array(y_test).ravel()
    predictions_arr = np.array([p[:,0] for p in predictions]).ravel()
    snrMax = float(np.max(snr, axis=0))
    snrMin = float(np.min(snr, axis=0))
    snrDict, lenSNRdict = createSNRDict(snrMax, snrMin, ranges)

    #each nested list's index corresponds to the output of snrDet when using the index as a param
    #ex the first index --> 0 corresponds to snrDet(0) == SNR [0-4]
    pred = []
    yarr = []
    snrs = []
    sorted_indicies = [] #snr indicies
    for i in range(lenSNRdict):
        pred.append([])
        yarr.append([])
        snrs.append([])
        sorted_indicies.append([])



    for i in range(predictions_arr.size):
        assert labels[test_indices[i]] == int(y_test_arr[i]), f"{labels[test_indices[i]]} is not equal to {int(y_test_arr[i])}. Check your indexing"
        index = int(snr[test_indices[i]]//(ranges))
        pred[index].append(predictions_arr[i])
        yarr[index].append(y_test_arr[i])
        snrs[index].append(snr[test_indices[i]])
        sorted_indicies[index].append((test_indices[i]))


    plotCMHeatmap(confusion_matrix(y_test_arr, predictions_arr), "Overall Confusion Matrix", states)
    
    for i in range(lenSNRdict):
        currCM = confusion_matrix(yarr[i], pred[i])
        if (len(currCM)!= 0):
            plotCMHeatmap(currCM, snrDict[i], states)
    

    






def trainingCurve(train_counter, train_losses, test_counter, test_losses):
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()




import matplotlib.pyplot as plt
import numpy as np
import time
from astropy import units as u
from Outside_Code.burst_sim import *
from Finished_Code.runCNN import *
import math
from scipy.special import erfinv


def plot3x4SetofNumbers(indiciesArr, snrRange, dataset, yarr, snrs, pred):
    for i in range(len(indiciesArr)):
        snrIndex = indiciesArr[i]
        plt.subplot(3,4,i+1)
        plt.imshow(dataset[snrIndex], cmap='gray', interpolation='none')
        plt.xlabel("Ground Truth: {}".format(int(yarr[snrRange][i])))
        plt.ylabel(f"SNR: {snrs[snrRange][i]:.2f}")
        plt.title("Prediction: {}".format(pred[snrRange][i]))
        plt.xticks([])
        plt.yticks([])
        
        
def pltFigsWithSnrs(numFigs, dataset, labels, snr):
    for i in range(numFigs):
        plt.imshow(dataset[i])
        plt.title(f'{labels[i]}, {snr[i]}')
        plt.show()


def pltNumTestFigsWithLabels(numFigs, data, targets):
    assert numFigs < data.size 
    fig = plt.figure()
    for i in range(numFigs):
        plt.subplot(2,5,i+1)
        plt.tight_layout()
        plt.imshow(data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(targets[i]))
        plt.xticks([])
        plt.yticks([])
        plt.show()

def jimFunc2002(dm, tauScatter = 0.65):
    logtau = -3.72 + (0.411 * np.log10(dm)) + (0.937 * (np.log10(dm)**2))
    noise = np.random.normal(scale = tauScatter)
    tau = 10**((logtau + noise)) * u.us
    return tau

def plotDMvsTau(dmlis, taulis):
    dmlis = np.array(dmlis).flatten()
    plt.scatter(dmlis, taulis)
    plt.xlabel("dm")
    plt.ylabel("tau")
    plt.show()



class Stopwatch:
    def __init__(self):
        self.checkpoint = 0
        self.timeDict = {}

    def createCheckpoint(self):
        self.currTime = time.time()
        self.timeDict[self.checkpoint] = self.currTime
         
        self.checkpoint+=1
        
    
    def findWorstCheckpoint(self):
        lastPoint = self.timeDict[0]
        maxDiff = 0
        worstKey = 0
        for i in range(self.checkpoint):
            currDiff = self.timeDict[i]-lastPoint
            if (currDiff > maxDiff):
                currDiff = maxDiff
                worstKey = i
            lastPoint = self.timeDict[i]
        return worstKey
    
    def findTimeDiff(self, key2, key1 = 0):
        return self.timeDict[key2] - self.timeDict[key1]

def exPlots(trial):
    # Here's some code for plotting the different frequency-averaged time series for different smearing effects
    plt.plot(trial.time_vec, trial.pulse, label=rf'Intrinsic Pulse $W_g$={trial.W_g}')
    plt.plot(trial.time_vec, con(trial.pulse[None,:], trial.smear_transfer).real.mean(axis=0), label=f'Intra-channel Smearing with DM={trial.DM} pc/cc')
    plt.plot(trial.time_vec, con(trial.pulse[None,:], trial.delay_transfer).real.mean(axis=0), label=rf'Inter-channel Smearing with $\delta$DM={trial.dDM} pc/cc')
    plt.plot(trial.time_vec, con(trial.pulse[None,:], trial.scatr_transfer).real.mean(axis=0), label=f'Scattered with $\\tau_{{sc}}$={trial.tau} at {trial.tau_rf.to(u.GHz)}')
    plt.plot(trial.time_vec, con(trial.pulse[None,:], trial.total_transfer).real.mean(axis=0), label='All of the Above')
    plt.legend()

    plt.xlim(-0.005, 0.015)

    plt.ylabel(r'$\frac{{S/N\;SEFD}}{S_\nu}=\langle\epsilon_i\rangle_\nu\sqrt{2 N_t N_\nu}\left(\bar{f}*m_{W_b}\right)(t)$', fontsize=16)
    plt.xlabel('Time [s]')
    plt.title(rf'f$_{{cent}}={trial.reference.to(u.MHz).value:0.0f}$MHz   B={trial.bandwidth.to(u.MHz).value:0.0f}MHz   N$_{{chan}}$={trial.N_chan}  Time Resolution={(1/trial.sample_rate).to(u.us).value}$\mu$s')

    plt.show()
    # Here's some code for plotting the dynamic spectrum of the pulse with all smearing effects
    plt.pcolormesh(trial.time_vec.value, trial.chan_centers.value, con(trial.pulse[None,:], trial.total_transfer).real)
    print(con(trial.pulse[None,:], trial.total_transfer).real)
    plt.show()

def runTest():
    trialin = PulseDistortion(bandwidth=250*u.MHz, reference=(1530+1280)*u.MHz/2, N_chan=2048, N_samples = 2**12)

    # Inject an intrinsic gaussian burst
    W_g = 1*u.ms # the gaussian has FWHM=1 ms
    trialin.add_pulse(W_g=W_g, SN=1)

    #metadata {'bandwidth': <Quantity 250. MHz>, 'reference': <Quantity 1405. MHz>, 'N_samples': 4096, 'sample_rate': <Quantity 0.12207031 MHz>, 'DM': 438.4361998910227, 'dDM': 2, 'tau': <Quantity 7.65347321e+28 ms>, 'tau_rf': <Quantity 1. GHz>}
    # Adding in the smearing effects from DM, error in DM, and scattering
    trialin.transfers(DM=1000, dDM=2, tau=100*u.ms, tau_rf=1*u.GHz) # these are just example values
    meta, shape, pulse = trialin.combine()
    print(meta)
    exPlots(trialin)


def jic():
    def createFinalTensor(bursts, nonBursts):
        #prep params
        datasetnp = np.array(bursts + nonBursts) 
        dataset_tensor = torch.from_numpy(datasetnp)
        labelsList = np.array([1] * len(bursts) + [0] * len(nonBursts))
        return dataset_tensor, labelsList


    def createNoisePP(data, SNRgoal, nChan= 2048):
        print("DATA")
        print(data)
        print(type(data.value))
        s_max = data.value.mean(axis = 0)
        noisePP = math.sqrt(nChan) * (s_max/SNRgoal)
        print("FINAL noisePP")
        print(noisePP)
        return noisePP

    def jimFunc2022(dm):
        tauScatter = np.random.normal(0.76, 1) 
        tau = 1.9e-7 * (dm**1.5) * (1 + 3.55e-5 * (dm**3.0))
        ret = u.ms.to(u.us, tau * 10**tauScatter) * u.us
        print(f"Jimfunc output {ret}")
        return ret

    def calcCDF(fwhm, p=0.001):
        sigma = sigmaCalc(fwhm)
        t = erfinv(1-p) * math.sqrt(2) * sigma
        print(f"CDF VALUE: {t}")
        print(f"CDF VALUE: {t.to(u.us)}")
        return t

    def sigmaCalc(fwhm):
        return fwhm/(2 * (2 * np.log(2)) ** (1/2))

    def calcTSLength(fwhm, freqMin, DMmax = 1000, p=0.001, scatterMean = 0.76):
        logTau = jimFunc2022(DMmax) + sigmaCalc(fwhm) * 3 * scatterMean #CHECKSCATTERMEAN
        tauSc = np.exp(logTau)
        tauFreqMin = tauSc*freqMin**(-4)
        finT = np.log((1/p))*tauFreqMin
        print(f"calcTSLength {finT}")
        return finT
        





    def plotDynamicSpectrum(par1, par2, par3, currdm, fwhm):
        #par1 = np.roll(burst.time_vec.to(u.us).value, -rollVal)
        
        cdf = calcCDF(fwhm) 
        
        #par3 = np.roll((con(burst.pulse[None,:], burst.total_transfer).real).value, -rollVal, axis = 0)
        plt.pcolormesh(par1, par2, par3) #time in mic secs
        plt.title(f"Dynamic Spectrum for DM: {currdm}")
        plt.xlabel("time (us)") 
        plt.ylabel("frequency (MHz)")
        plt.colorbar()
        plt.show()

        avged_par3 = np.mean(par3, axis = 0)
        plt.scatter(par1, avged_par3)
        plt.title(f"Freq avg by time: {currdm}")
        plt.xlabel("time (us)") 
        plt.ylabel("frequency (MHz)")
        plt.show()



    def genOneSim(dm:float, dDM:float, tau:u.Quantity, rollVal, snrGoal, w_g = 1*u.ms, n_samples = 2**12, n_chan=2048, bandwidth = 250*u.MHz, reference=(1530+1280)*u.MHz/2, snr = 1, test = False):

        """
        Required params: 
        DM
        dDM
        tau
        tau_rf --> 1 GHz
        W_g --> is this btwn 1-10 us or btwn 10-100 ms?
        bandwidth
        reference 
        N_chan
        N_samples
        SNR --> 1
        """
        tau_rf = 1 * u.GHz #(1P)
        burst = PulseDistortion(bandwidth = bandwidth, reference=reference, N_chan=n_chan, N_samples=n_samples) # is the "<param> = {value}" necessary? (5Q)
        burst.add_pulse(W_g=w_g, SN=1)

        # Adding in the smearing effects from DM, error in DM, and scattering
        burst.transfers(DM=dm, dDM=dDM, tau=tau, tau_rf=tau_rf) 

        # Output of combining the intrinsic pulse with these effects: metadata for this simulation, the intrinsic shape, and the smeared shape
        meta, shape, pulse1 = burst.combine() #would this be useful to return somewhere? (6Q)
        
        plt.show()
        par1 = burst.time_vec.to(u.us).value
        par2 = burst.chan_centers.to(u.MHz).value
        outputBefSNRPP = con(burst.pulse[None,:], burst.total_transfer).real

        ppnVal = createNoisePP(outputBefSNRPP, snrGoal)
        print(meta)
        meta["SNRfinal"] = snrGoal
        print(meta)
        filters = np.random.exponential(ppnVal, size = outputBefSNRPP.shape)
        #burstwSNR = burst.addPPNoise(ppnVal)
        
        output = outputBefSNRPP + filters
        print(f"LOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOK {rollVal}")
        par3 = np.roll(output.value, 0, axis = 1)
        
        #plotDynamicSpectrum(burst, dm)
        return par1, par2, par3, meta

    ##find signal peak before tensor packaging
    def calcdDM(dm, refPath):
        f = open(refPath, 'r')
        dm_t_str = f.read().split()
        dm_t = np.array([float(s) for s in dm_t_str])
        print(dm_t_str)
        print(dm_t)
        f.close()
        dm_n = dm - dm_t
        print(dm_n)
        dDM = min(abs(dm_n))
        print(dDM)
        return dDM

    #ADD TXT FILE AS A PARAM BELOW
    def createDSDataset(noBursts:int, plotDM:bool=False):


        dm = np.random.uniform(2,1000, noBursts) 
        snrGoals = np.random.uniform(0.1,10, noBursts) 
        
        taulis = []
        dataset = []
        metadatalis = []
        
        w_g = 0.05*u.ms 

        for i in range(noBursts):
            dDM = calcdDM(dm[i], 'dmlist.txt') 
            test = False
            currdm = dm[i]
            tau = jimFunc2022(currdm)
            taulis.append(tau.value)
            if (i == 0):
                test = True

            rollVal = 1870
            print("LOOOOOOOOOOOOOOOOOOOOOOOOOOOOK {rollVal}")
            par1, par2, par3, metadata = genOneSim(currdm, dDM, tau, rollVal, snrGoals[i], w_g = w_g, test=test)


            if (plotDM): 
            
                plotDynamicSpectrum(par1, par2, par3, currdm, w_g) #change to negative input for param 1
            
            dataset.append(par3)
            metadatalis.append(metadata) #ADD SNR TO METADATA

            
        plotDMvsTau(dm, taulis)

        return dataset, metadatalis





    data, metadataRET = createDSDataset(2, plotDM=True)
    #data, metadataRET = createDSDataset(11)


def plotHistogram(filters, snr):
    filtermeans = 1/np.mean(filters, axis = (1,2))
    plt.hist(filtermeans, bins = 100, histtype = "step")
    plt.hist(snr, bins = 100, histtype = "step")
    plt.show()




##########################################################################################################
#MAKE ADJUSTMENTS TO PARAMS AND CHECK FUNCTION TO MAKE SURE IT ACTUALLY WORKS BC UNTESTED EDITS WERE MADE#
##########################################################################################################

def runCNNwFilters(dataset_in, labels, n_epochs, batch_size_train, batch_size_test, n_train, snr, states, 
            out_channel1, out_channel2, final_out, kernel_size_conv, kernel_size_pool, reshape_dims, fc_dims, 
            learning_rate = 0.01, momentum = 0.5, cm = False, ranges= None, training_Curve = False):
    """Runs the entire neural network with options to plot different types of plots if needed.

    Args:
        dataset_in: a tensor that contains all of the data required for training and testing the neural network
        labels: a np array of all the ground truth categorizations
        n_epochs: an integer to determine number of train and test cycles needed
        batch_size_train: an integer to determine what the size of each training batch should be
        batch_size_test: an integer to determine what the size of each test batch should be
        n_train: the number of tensors of data that will be trained (the remainder will go towards testing)
        snrMax: an integer or float to set the maximum signal to noise ratio included in the dictionary
        snrMin: an integer or float to set the minimum signal to noise ratio included in the dictionary
        ranges: an integer or float to set the amount of signal to noise in each category of the dictionary
        states: the classification states
        out_channel1: the number of channels going out of the first convolutional layer (a large increase- ex. 3 to 50- is reccommended)
        out_channel2: the number of channels going out of the second convolutional layer
        final_out: the out_features from the second fully connected layer
        kernel_size_conv: An integer to set the dimesions of the square kernel for the convolutional layers
        kernel_size_pool: An integer to set the dimesions of the square kernel for the pooling layers
        reshape_dims: An integer that sets the in_features of the first fully connected layer
        fc_dims: An integer that sets the out_features of the first fully connected layer
        learning_rate: float or integer that is used to determine the learning rate of the optimizer
        momentum: a float or integer that is used to determine the momentum of the optimizer
        cm: a boolean that determines if the confusion matrices of the determined snr ranges will be plotted
        ranges: an integer or float to set the amount of signal to noise in each category of the dictionary
            note: if cm is True ranges must be given a value
        trainingCurve: a boolean that determines if the training curve for the network will be plotted

    """
    
    #prep params
    logger.info("begin runCNN")
    dataset = dataset_in.numpy()

    dataset = dataset / dataset.max(axis=(1,2))[:,None, None]


    #define new lcl vars
    y_test = []
    predictions = [] 
    logger.debug(f"labels shape {labels.shape[0]}")
    #indices
    indices = np.array(range(labels.shape[0]))
    logger.debug(f"indicies {indices}")
    np.random.shuffle(indices)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    #split datasets
    train_dataset = dataset[train_indices]
    test_dataset = dataset[test_indices]
    train_labels = labels[train_indices]
    test_labels = labels[test_indices]

    #create filters
    snr = np.random.uniform(snrMin,snrMax,dataset_in.shape[0])
    filters = np.random.exponential((1/snr[:,None,None]), dataset.shape)

    #split filters
    test_filters = filters[test_indices]
    train_filters = filters[train_indices]

    #add filters to dataset
    train_dataset += train_filters 
    test_dataset += test_filters
    
    #temp print statements
    
    #convert to DataLoaders
    tensor_train = torch.Tensor(train_dataset[:,None, :, :])
    tensor_test = torch.Tensor(test_dataset[:, None, :, :])
    tensor_train_labels = torch.Tensor(train_labels)
    tensor_test_labels = torch.Tensor(test_labels)
    dataset_train = TensorDataset(tensor_train, tensor_train_labels)
    dataset_test = TensorDataset(tensor_test, tensor_test_labels)
    train_loader = DataLoader(dataset_train, batch_size = batch_size_train)
    test_loader = DataLoader(dataset_test, batch_size = batch_size_test)

    #create network + tracker vars
    network = Net(tensor_train.shape[1], out_channel1, out_channel2, final_out, kernel_size_conv, kernel_size_pool, reshape_dims, fc_dims)
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum = momentum)
    train_losses = []
    train_counter = []
    test_losses = []
    

    #run network
    test_losses, y_test, predictions = test(network, test_loader, test_losses)
    for epoch in range(1, n_epochs + 1): 
        train_losses, train_counter = train(epoch, network, train_loader, optimizer, train_losses, train_counter)
        test_losses, y_test, predictions = test(network, test_loader, test_losses)
        


    if(training_Curve):
        test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]
        trainingCurve(train_counter, train_losses, test_counter, test_losses)
    
    logger.debug(f"y_test, {y_test}")
    logger.debug(f"predictions {predictions}")
    if(cm):
        assert ranges is not None
        makeAllCMPlots(ranges, y_test, predictions, labels, test_indices, snr, states)
    
    


def runMNIST():
    dataset1 = torchvision.datasets.MNIST('/files/', download=True).data
    print(dataset1.shape)
    print(type(dataset1))
    labels1 = torchvision.datasets.MNIST('/files/', download=True).targets.numpy()
    print(labels1.shape)
    snrMax = 10
    snrMin = 0.1
    num_epochs = 3
    batch_size_train1 = 64
    batch_size_test1 = 1000
    num_train = 50000
    ranges = 2
    states = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    kernel_size_conv = 5
    kernel_size_pool = 2
    out_channel1 = 25
    out_channel2 = 2
    final_out = 10
    numLayers = 2
    snrs = np.random.uniform(1,10, labels1.shape[0])
    for i in range(2):
        H_out, W_out = computeNextDimsCNN(dataset1.shape[2], dataset1.shape[2], kernel_size_conv, padding = (0,0), dilation = (1,1), kernel_size_pool = 2, numLayers = 1)
        print(f"shape is {batch_size_train1}, 1, {H_out}, {W_out}")
    #print(computeMatrixReqs(dataset1.shape[2], dataset1.shape[2], kernel_size_conv, batch_size_test1, out_channel2, kernel_size_pool, numLayers))
    fc_dims, reshape_dims = defDims(dataset1.shape[2], dataset1.shape[2], kernel_size_conv, batch_size_test1, out_channel2, kernel_size_pool=kernel_size_pool, numLayers = numLayers)
    runCNN(dataset1, labels1, num_epochs, batch_size_train1, batch_size_test1, num_train, snrs, states,
            out_channel1, out_channel2, final_out, kernel_size_conv, kernel_size_pool, reshape_dims, fc_dims, cm=True, ranges= ranges, training_Curve=True)

    


def runCNN_NoFilters(dataset_in, labels, snr, n_epochs, batch_size_train, batch_size_test, n_train, ranges, states, 
            out_channel1, out_channel2, final_out, kernel_size_conv, kernel_size_pool, reshape_dims, fc_dims, 
            learning_rate = 0.01, momentum = 0.5, cm = True, training_Curve = False):
    """Runs the entire neural network with options to plot different types of plots if needed.

    Args:
        dataset_in: a tensor that contains all of the data required for training and testing the neural network
        labels: a np array of all the ground truth categorizations
        snr: a np array that describes the signal to noise ratio (indices should correspond to dataset_in)
        n_epochs: an integer to determine number of train and test cycles needed
        batch_size_train: an integer to determine what the size of each training batch should be
        batch_size_test: an integer to determine what the size of each test batch should be
        n_train: the number of tensors of data that will be trained (the remainder will go towards testing)
        ranges: an integer or float to set the amount of signal to noise in each category of the dictionary
        states: the classification states
        out_channel1: the number of channels going out of the first convolutional layer (a large increase- ex. 3 to 50- is reccommended)
        out_channel2: the number of channels going out of the second convolutional layer
        final_out: the out_features from the second fully connected layer
        kernel_size_conv: An integer to set the dimesions of the square kernel for the convolutional layers
        kernel_size_pool: An integer to set the dimesions of the square kernel for the pooling layers
        reshape_dims: An integer that sets the in_features of the first fully connected layer
        fc_dims: An integer that sets the out_features of the first fully connected layer
        learning_rate: float or integer that is used to determine the learning rate of the optimizer
        momentum: a float or integer that is used to determine the momentum of the optimizer
        cm: a boolean that determines if the confusion matrices of the determined snr ranges will be plotted
        trainingCurve: a boolean that determines if the training curve for the network will be plotted
    """
    
    #prep params
    if(not(isinstance(np.ndarray))):
        dataset = dataset_in.numpy()

    dataset = dataset / dataset.max(axis=(1,2))[:,None, None]
    

    #define new lcl vars
    y_test = []
    predictions = [] 
    snrMax = snr.max()
    snrMin = snr.min()
    

    #indices
    indices = np.array(range(labels.shape[0]))
    print(indices)
    np.random.shuffle(indices)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    #split datasets
    train_dataset = dataset[train_indices]
    test_dataset = dataset[test_indices]
    train_labels = labels[train_indices]
    test_labels = labels[test_indices]

    
    #convert to DataLoaders
    tensor_train = torch.Tensor(train_dataset[:,None, :, :])
    tensor_test = torch.Tensor(test_dataset[:, None, :, :])
    tensor_train_labels = torch.Tensor(train_labels)
    tensor_test_labels = torch.Tensor(test_labels)
    dataset_train = TensorDataset(tensor_train, tensor_train_labels)
    dataset_test = TensorDataset(tensor_test, tensor_test_labels)
    train_loader = DataLoader(dataset_train, batch_size = batch_size_train)
    test_loader = DataLoader(dataset_test, batch_size = batch_size_test)

    #create network + tracker vars
    network = Net(tensor_train.shape[1], out_channel1, out_channel2, final_out, kernel_size_conv, kernel_size_pool, reshape_dims, fc_dims)
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum = momentum)
    train_losses = []
    train_counter = []
    test_losses = []
    

    #run network
    test_losses, y_test, predictions = test(network, test_loader, test_losses)
    for epoch in range(1, n_epochs + 1): 
        train_losses, train_counter = train(epoch, network, train_loader, optimizer, train_losses, train_counter)
        test_losses, y_test, predictions = test(network, test_loader, test_losses)


    if(training_Curve):
        test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]
        trainingCurve(train_counter, train_losses, test_counter, test_losses)
    

    if(cm):
        makeAllCMPlots(ranges, y_test, predictions, snrMax, snrMin, labels, test_indices, snr, states)
    
    

def lossyDownsample(par1):
    
    length = len(par1)
    if(length < 1000):
        return par1
    else:
        downsamplepd = length//1000
        down_par1 = par1[0::downsamplepd]
    print(f"length par1 bef {len(par1)}")
    print(f"length par1 after {len(down_par1)}")
    print(f"lossdown_par1 {down_par1}")
    return down_par1
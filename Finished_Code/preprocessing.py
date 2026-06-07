from Outside_Code.burst_sim import *
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
import math
from scipy.special import erfinv
import torchvision 
import logging
import torch.nn.functional as F


def avgDownsample(par1, downsizeFact):

    cutFact = math.floor((20*u.ms).to(u.us).value / (8.2 *u.us).value)
    length = len(par1)
    if(length < 1000):
        return par1
    
    else:
        div = findClosestDiv(length, downsizeFact) 

        split_par1 = np.reshape(par1, (length//div, div))
        down_par1 = np.mean(split_par1, axis = 1).flatten()
        newlen = len(down_par1)
        if(newlen > cutFact):
            i = (newlen - cutFact)//2
            return down_par1[i:newlen-i]

        return down_par1

def findClosestDiv(n, x):
    if(n%x == 0):
        return x
    i = 0
    while(True):
        if(n%(x-i)==0):
            return x-i
        elif(n%(x+i) == 0):
            return x+i
        i+=1



def createNoisePP(data, SNRgoal, nChan= 2048):

    s_max = data.value.mean(axis = 0)
    noisePP = math.sqrt(nChan) * (s_max/SNRgoal)
    return noisePP

def jimFunc2022(dm):
    tauScatter = np.random.normal(0.76, 1) 
    tau = 1.9e-7 * dm**1.5 * (1 + 3.55e-5 * dm**3.0)
    return u.ms.to(u.us, tau * 10**tauScatter) * u.us

def calcCDF(fwhm, p=0.001):
    sigma = sigmaCalc(fwhm)
    t = erfinv(1-p) * math.sqrt(2) * sigma
    return t

def sigmaCalc(fwhm):
     return fwhm/(2 * (2 * np.log(2)) ** (1/2))



def calcTSLength(freqMin= 1280.06103515625, DMmax = 400, p=0.01, scatterMean = 0.76):

    tau = 1.9e-7 * DMmax**1.5 * (1 + 3.55e-5 * DMmax**3.0)

    tau_dm = tau * (10**scatterMean* 3 ) 
    tauFreqMin = tau_dm*((freqMin*1e-3)**(-4))
    finT = np.log((1/p))*tauFreqMin*u.ms

    return finT.to(u.us)


def plotDynamicSpectrum(par1, par2, par3, currdm):


    
    avged_par3 = np.mean(par3, axis = 0)
    plt.scatter(par1, avged_par3)
    plt.title(f"Freq avg by time: {currdm}")
    plt.xlabel("time (us)") 
    plt.ylabel("frequency (MHz)")
    plt.show()

    
    #par3 = np.roll((con(burst.pulse[None,:], burst.total_transfer).real).value, -rollVal, axis = 0)
    plt.pcolormesh(par1, par2, par3) #time in mic secs
    plt.title(f"Dynamic Spectrum for DM ROLL: {currdm}")
    plt.xlabel("time (us)") 
    plt.ylabel("frequency (MHz)")
    plt.colorbar()
    plt.show()


def genOneSim(dm:float, dDM:float, tau:u.Quantity, snrGoal, w_g = 1*u.ms, n_chan=256, bandwidth = 250*u.MHz, reference=(1530+1280)*u.MHz/2, test = False):

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
    samp_rate = 1/(8.2*u.us) 
    lengthTimeAxis = calcTSLength()
    n_samples = math.ceil(lengthTimeAxis.value * samp_rate.value)
    

    #find max n_samples required for DM 1000
    burst = PulseDistortion(bandwidth = bandwidth, reference=reference, N_chan=n_chan, N_samples=n_samples, sample_rate = samp_rate) # is the "<param> = {value}" necessary? (5Q)
    burst.add_pulse(W_g=w_g, SN=1)

    # Adding in the smearing effects from DM, error in DM, and scattering
    burst.transfers(DM=dm, dDM=dDM, tau=tau, tau_rf=tau_rf) 

    # Output of combining the intrinsic pulse with these effects: metadata for this simulation, the intrinsic shape, and the smeared shape
    meta, shape, pulse1 = burst.combine() 
    
    
    par1 = burst.time_vec.to(u.us).value
    par2 = burst.chan_centers.to(u.MHz).value
    outputBefSNRPP = con(burst.pulse[None,:], burst.total_transfer).real
    down2_par1 = avgDownsample(par1, 4)
    ppnVal = createNoisePP(outputBefSNRPP, snrGoal)
    meta["SNR"] = snrGoal



    filters = np.random.exponential(ppnVal, size = outputBefSNRPP.shape)
    #burstwSNR = burst.addPPNoise(ppnVal)
    
    output = outputBefSNRPP + filters
    #plotDynamicSpectrum(burst, dm)
    return output, down2_par1, par2, meta

##find signal peak before tensor packaging
def calcdDM(dm, refPath):
    f = open(refPath, 'r')
    dm_t_str = f.read().split()
    dm_t = np.array([float(s) for s in dm_t_str])
    f.close()
    dm_n = dm - dm_t
    dDM = min(abs(dm_n))
    return dDM

#ADD TXT FILE AS A PARAM BELOW
def createDSDataset(noBursts:int, plotDM:bool=False):


    dm = np.random.uniform(2,1000, noBursts) 
    snrGoals = np.random.uniform(1,10, noBursts) 
    
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

        data, par1, par2, metadata = genOneSim(1000, dDM, tau, snrGoals[i], w_g = w_g, test=test)

        cdf = calcCDF(w_g) 

        t = cdf.to(u.us).value

        samp_rate = 8.2*u.us

        K = 4.15e3
        fm = np.max(par2)
        fc = np.median(par2)
        correc_fact = dDM * ((1/(fm))**2 - (1/(fc))**2) * K

        dt =abs(par1[0]) - t + abs(correc_fact) 

        dt_ind =  math.floor(dt/samp_rate.value)

        par3 = np.roll((data).value, -dt_ind, axis = 1)


        if (plotDM): 
           plotDynamicSpectrum(par1, par2, par3, currdm) #change to negative input for param 1
        
        dataset.append(data)
        metadatalis.append(metadata) #ADD SNR TO METADATA


    return dataset, metadatalis, snrGoals

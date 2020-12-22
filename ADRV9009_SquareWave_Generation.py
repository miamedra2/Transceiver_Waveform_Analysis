import iio
import adi
import pandas as pd
import scipy
import scipy.fftpack
from peakdetect import peakdetect
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import signal
from scipy.fftpack import fft
from numpy.fft import fftshift, ifft
import h5py
import math as m

###############################################################################
# Class contains the methods to make an FFT of the waveforms produced by GenerateSinewaves.
# Input is F_in the input frequency, F_sample the sample rate of the time domain 
# data, bits the number of bits that the signal has.
###############################################################################
class Quantization_Spur_Analysis:
    
    F_in = 0 
    F_sample = 0
    bits = 0
    N = 0   
    
    ###############################################################################
    # Constructor of Squre_Wave sets the global fields
    ###############################################################################
    def __init__(self, F_in = 10e6, F_sample = 245.76e6, bits = 16, N = 1024):
        
        self.F_in = F_in
        self.F_sample = F_sample
        self.bits = bits
        self.N = N

    ###############################################################################
    # Here data is just an array of the complex data np.re +1j*np.imag, F_sample is the sample frequency, 
    # N_t is FFT size and B is the number of bits.  This FFT method is fit only for periodic data- so
    # don't use it for the chirps.
    ###############################################################################
    def getFFT_noise(self, data, N_t = 1024, plot_en = True):
        
        sample_rate = self.F_sample 
        D = self.bits
        
        N = len(data)

        N_a = int(np.floor(4*N/N_t)-3)
        R_xx = np.zeros(int(N_t))
        w = np.kaiser(N_t,11)
        w_cg = np.mean(w)
        w_ig = np.sqrt(np.mean(w**2))
        B_max=16
    
        #print("FFT Size: "+ str(N_t) + " Averages: " + str(N_a))

        for i in range(N_a):
            i_w = int(i*N_t/4)
            x_est = data[i_w:N_t+i_w]
            X_est = scipy.fftpack.fft(w*x_est)
            R_xx = R_xx + np.abs(X_est)**2
    
        R_xx = R_xx/(N_a*(N_t)**2*(2**(B_max-1))**2*(w_cg**2))
        xf = np.arange(-int(N_t/2), int(N_t/2))/N_t*sample_rate/1e6
        R_dB = np.fft.fftshift(10*np.log(R_xx)/np.log(10))
    
        #R_out = np.append(R_xx[int(N_t/2):], R_xx[:int(N_t/2)])
    
        if(plot_en == True):
            plt.figure(figsize=(6,6))
            plt.axhline(y = -(6.02*D-2.32), color = 'red', label = "SFDR Bound")
            plt.axhline(y = -(6.02*D+1.76+10*np.log(N/2)/np.log(10)+20*np.log(w_cg/w_ig)/np.log(10)), color = 'green', label = "Quantization Noise")
            plt.plot(xf, R_dB, label = "Data")
            plt.xlabel("Frequency (MHz)")
            plt.ylabel("Power (dbFS)")
            plt.grid(True)
            plt.legend()
            plt.title("")
            plt.show()
    
        return xf, R_dB, w_cg, w_ig
    
###############################################################################
# This class is used to Generate Sinewaves and quantizes them to [-32767, 32767]. 
# F_in is the input frequency, F_sample is the sample rate of the transceiver, 
# N is the number of samples to generate.  This class has a method __get_fin(),
# which obtains F_actual from F_in.  F_actual is used to generate the tone in 
# __make_waveform().
###############################################################################
class GenerateSinewaves:
    
    F_in = 0 
    F_sample = 0
    bits = 0
    N = 0
    F_actual = 0
    channel_spacing = 0
    M = 0
    
    ###############################################################################
    # Constructor of Squre_Wave sets the global fields
    ###############################################################################
    def __init__(self, F_in = 10e6, F_sample = 245.76e6, bits = 16, N = 256000):
        self.F_sample = F_sample
        self.bits = bits
        self.N = N
        self.F_in = F_in
        self.F_actual = self.__get_fin(F_in, F_sample, N)
    
    ###############################################################################
    # returns a squarewave corresponding to F_in
    ###############################################################################
    def __get_onebitsine(self, F_in, F_sample):
        squarewave_i = []
        squarewave_q = []
    
        N_2pi = F_sample/F_in
        endval = int(256000/N_2pi)*N_2pi
        amplitude = 32767
    
        sample_index = np.arange(0, endval)
    
        data_i = np.cos((2*np.pi*sample_index)/N_2pi)
        data_q = np.sin((2*np.pi*sample_index)/N_2pi)
    
        squarewave_i = [amplitude*2*(np.round(elem)-.5) for elem in 0.5*(data_i+1)]
        squarewave_q = [amplitude*2*(np.round(elem)-.5) for elem in 0.5*(data_q+1)]

        return np.array(squarewave_i), np.array(squarewave_q)
    
    ###############################################################################
    # returns a quantized sinewave corresponding to F_in.  bits can vary between 1-16 corresponding to 2 bit
    # and so on DAC.  For one bit dac use the _get_onebitsine method.
    ###############################################################################
    def __get_Nbitsine(self, F_in, F_sample, bits):
    
        N_2pi = F_sample/F_in
        endval = int(256000/N_2pi)*N_2pi
        amplitude = 32767
    
        sample_index = np.arange(0, endval)
    
        data_i = np.cos((2*np.pi*sample_index)/N_2pi)
        data_q = np.sin((2*np.pi*sample_index)/N_2pi)
    
        quant = max(data_i)/(2**bits-1)
    
        data_i = np.round(data_i/quant)
        data_q = np.round(data_q/quant)

        max_val = max(data_i)
    
        data_i = [int(amplitude*elem/max_val) for elem in data_i]
        data_q = [int(amplitude*elem/max_val) for elem in data_q]
    
        return np.array(data_i), np.array(data_q)
    
    ###############################################################################
    # Finds a frequency close to F_appox that will select all the ADC codes- based on the sample rate and the
    # number of data points.
    ###############################################################################
    def __get_fin(self, F_approx, F_sample, N):
        channel_spacing = F_sample/N
        M = 2*round(F_approx/(2*channel_spacing))+1
        F_out = M/N*F_sample
        
        print("M: " + str(M) +" N:" + str(N) + " Channel Spacing: " + str(channel_spacing) + " F_actual: " + str(F_out))
        
        self.channel_spacing = channel_spacing
        self.M = M
    
        return F_out
    
    ###############################################################################
    # gets a quantized sine data based on F_in.  Also plots if plot_en is True
    ###############################################################################
    def __make_waveform(self, plot_en = True):  
        
        F_in = self.F_actual
        F_sample = self.F_sample
        bits = self.bits
        
        if((bits-1) == 0):
            data_i, data_j =  self.__get_onebitsine(F_in, F_sample)  
        else:
            data_i, data_j =  self.__get_Nbitsine(F_in, F_sample, bits-1)
    
        if(plot_en==True):
            plt.figure(figsize = (6,6))
            plt.xlim(0,500)
            plt.plot(data_i, label = "Channel I")
            plt.plot(data_j, label = "Channel Q")
            plt.xlabel('Time (Samples)')
            plt.ylabel('Amplitude (ADC Code)')
            plt.legend()
            plt.title("Ideal Waveform")
            plt.show()
    
        return data_i, data_j
    
    ###############################################################################
    # makes a waveform corresponding to F_actual, F_sample, and N_sample
    ###############################################################################
    def make_ideal_waveform(self, verbose = False):
        data_i, data_q = self.__make_waveform(plot_en = verbose)
        
        spur_analysis = Quantization_Spur_Analysis(F_in = self.F_actual, F_sample = self.F_sample, bits = self.bits, N = self.N)
        xf, R_dB, w_cg, w_ig = spur_analysis.getFFT_noise(data = data_i+1j*data_q, N_t = 1024, plot_en = verbose)
        
        return xf, R_dB, data_i+1j*data_q
        
 
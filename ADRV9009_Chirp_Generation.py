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
# Square_wave redefines make_waveform in simple_chirp to create a chirp wave
# pattern.  It takes as input F_sample = the sample rate of the tranceiver, 
# N_sample is the number of samples which defaults to the size of the bram 40 bits*4096,
# F_center is the center frequency of the chirp, B_chirp is the chirp bandwidth, T_chirp
# is the chirp time duration, and text_name is where the txbram_data is saved to.
###############################################################################
class chirp_wave():
    
    F_sample = 0
    F_center = 0
    B_chirp = 0
    T_chirp = 0
    T_zeros = 0
    T_chirp_act = 0
    K_chirp = 0
    bits = 1
    
    ###############################################################################
    #Constructor of chirp_wave sets the global fields
    ###############################################################################
    def __init__(self, F_sample = 245.76e6, F_center = 0, B_chirp = 30e6, T_chirp = 20e-6, T_zeros = 55e-6, bits = 1):
        
        self.F_sample = F_sample
        self.F_center = F_center
        self.B_chirp = B_chirp
        self.T_chirp = T_chirp 
        self.T_zeros = T_zeros
        self.K_chirp = B_chirp/T_chirp
        self.bits = bits
    
    ###############################################################################
    # Given a list t_n, and chirp params K_chirp, and F_centers returns an array containing an unquantized chirp
    ###############################################################################
    def chirp_exp(self, t_n, K_chirp, F_center):
        return np.exp(1j*2*np.pi*(t_n*F_center+1/2*K_chirp*t_n**2))
    
    def make_waveform(self, verbose = True):
        #make a floating point chirp
        data_i, data_q, T_chirp_act = self.make_chirp()
        
        #put the actual chirp duration in the class vars
        self.T_chirp_act = T_chirp_act
        
        #quantize the chirp according to self.bits
        data_quant_i, data_quant_q = self.quantize_chirp(data_i = data_i, data_q = data_q)
        
        #pad the zeros according to T_zeros.  In this case adds T_zeros/2 0s to the front and half
        data_pad_i, data_pad_q, window = self.add_zeroes_to_chirp(data_i = data_quant_i, data_q = data_quant_q, verbose = verbose)
        
        return data_pad_i, data_pad_q, window
        
    ###############################################################################
    # Redefines make_waveform in simple_chirp to create a chirp based on the constructor arguments of chirp_wave.  
    # Saves the chirp_waveform by putting 40 bit chunks into the file specified by text_name.
    ###############################################################################
    def make_chirp(self):

        RX_sample_rate = self.F_sample
        T_chirp = self.T_chirp
        F_center = self.F_center
        B_chirp = self.B_chirp
    
        #finds the number of samples required to make the chirp - which may not 
        #correspond to T_chirp exactly
        N_sample_chirp = int(np.round(RX_sample_rate*T_chirp))
    
        #finds the actual chirp duration
        T_chirp_act = N_sample_chirp/RX_sample_rate
        print("Actual Chirp Duration: "+ str(T_chirp_act))
    
        #recalculates K_chirp
        K_chirp_act = B_chirp/T_chirp_act
    
        #creates a np array centered on 0 from -N_sample/2 to N_sample_2
        n = np.arange(-N_sample_chirp/2,N_sample_chirp/2)
        
        #divides the array by the sample rate to put in units of time
        t = n/RX_sample_rate
        
        #gets the chirp and makes an array in terms of x
        x = np.array([self.chirp_exp(elem[1], K_chirp_act, F_center) if abs(elem[1])<=T_chirp_act/2 else 0 for elem in enumerate(t)])
   
        #normalizes the chirp to 1 and then rounds (0,1) and casts each elem to an int to get a list   
        #data_i = [32767*2*(elem - 0.5) for elem in np.round(0.5*(np.real(x)+1))]
        #data_q = [32767*2*(elem - 0.5) for elem in np.round(0.5*(np.imag(x)+1))]
        
        #quant_data = np.array(data_i) + 1j*np.array(data_q)
    
        return np.real(x), np.imag(x), T_chirp_act
    
    def quantize_chirp(self, data_i, data_q):
    
        amplitude = 0xFFFF
        bits = self.bits
    
        if(bits <= 16 and bits > 1):
        
            bits = bits - 1
            quant = max(data_i)/(2**bits-1)
    
            data_i = np.round(data_i/quant)
            data_q = np.round(data_q/quant)

            max_val = max(data_i)
    
            data_quant_i = [int(amplitude*0.5*elem/max_val) for elem in data_i]
            data_quant_q = [int(amplitude*0.5*elem/max_val) for elem in data_q]
        elif(bits == 1):
            #normalizes the chirp to 1 and then rounds (0,1) and casts each elem to an int to get a list   
            data_quant_i = [amplitude*(elem - 0.5) for elem in np.round(0.5*(data_i+1))]
            data_quant_q = [amplitude*(elem - 0.5) for elem in np.round(0.5*(data_q+1))]
        else:
            print("Input bits should be between 1 and 16")
            return np.zeros(len(data_i)), np.zeros(len(data_i))
    
        return np.array(data_quant_i), np.array(data_quant_q)   
    
    def add_zeroes_to_chirp(self, data_i, data_q, verbose):
    
        RX_sample_rate = self.F_sample
        T_chirp = self.T_chirp
        T_zeros = self.T_zeros
        F_center = self.F_center
        B_chirp = self.B_chirp
    
        #finds the number of samples of zeroes to create
        N_samples_zeros = int(RX_sample_rate*T_zeros/2)
   
        #makes a list set to 0xFFFF with size N_samples_zeros
        zeros = [0]*N_samples_zeros
    
        #adjusting list elems is easier than dealing with arrays
        data_i = list(data_i)
        data_q = list(data_q)
    
        #pad the real/imag parts with zeros
        data_i_gap = zeros + data_i + zeros
        data_q_gap = zeros + data_q + zeros
    
        #creates windowing function
        window = list(signal.tukey(len(data_i), 0.5))
    
        window_gap = zeros + window + zeros
    
        if(verbose == True):
            #creates an np.array centered on 0 from -N_sample/2 to N_sample_2
            n = np.arange(-len(data_i_gap)/2,len(data_i_gap)/2)
        
            #divides the array by the sample rate to put in units of time
            t = n/RX_sample_rate
        
            plt.plot(t, data_i_gap)
            plt.plot(t, np.array(window_gap)*32767)
            plt.show()

        return np.array(data_i_gap), np.array(data_q_gap), np.array(window_gap)*32767
    
        
###############################################################################
# Gets an FFT of data.  chirp_spectrum_analysis should be initialized with the ADRV9009 ADC 
# sampling params F_sample - the sample rate and N_sample -  the number of samples in the 
# data set and chirp parameters F_center - the chirp center frequency, B_chirp - the bandwidth
# of the chirp and the T_chirp - the chirp duration.  The main methods used for FFTs
# in chirp_spectrum_analysis are get_power() and get_average_power().
###############################################################################
class chirp_spectrum_analysis:

    F_sample = 0
    F_center = 0
    B_chirp = 0
    T_chirp = 0
    K_chirp = 0
    T_chirp_act = 0
    
    ###############################################################################
    # constructor takes args and writes to class vars.
    ###############################################################################
    def __init__(self, F_sample, F_center, B_chirp, T_chirp):
        
        self.F_sample = F_sample
        self.F_center = F_center
        self.B_chirp = B_chirp
        self.T_chirp = T_chirp 
        self.K_chirp = B_chirp/T_chirp
        
    ###############################################################################
    # returns the global params 
    ###############################################################################
    def get_chirp_params(self):
        
        return self.F_center, self.B_chirp, self.F_sample, self.T_chirp, self.K_chirp
        
    ###############################################################################   
    # returns a normalized fft of data*window and the freq x-axis in units of MHz.  data is
    # assumed to be a list of time series data, F_sample is the sample rate, and window is a 
    # list with same size as len(data)
    ###############################################################################
    def get_chirp_powerspectrum(self, data, F_sample, window, verbose=False):
        
        F_center, B_chirp, F_sample, T_chirp, K_chirp = self.get_chirp_params()
              
        #multiply data by window elem by elem
        product = [data[i]*window[i] for i in range(len(data))]
        
        if(verbose):
            plt.figure(figsize=(7, 6))
            plt.grid()
            plt.xlabel("Number of Samples")
            plt.ylabel("Amplitude (arb)")
            plt.plot(np.real(data), label = "real ideal data")
            plt.plot(window, label = "windowing function")
            plt.legend()
            plt.show()
            
        #gets N_sample from the length of data
        N_sample = len(data)

        #get the fft of the product
        X = np.fft.fftshift(scipy.fftpack.fft(np.fft.fftshift(product))) 
        
        #normalize the fft to 2**16/2-1
        X_norm = (abs(X)*np.sqrt(K_chirp)/(F_sample*32767))**2
        
        #create frequency range according to the number of samples and sample rate
        if(N_sample % 2  == 0):
            xf = np.arange(-int(N_sample/2), int(N_sample/2))/N_sample*F_sample/1e6
        else:
            xf = np.arange(-int(N_sample/2), int(N_sample/2)+1)/N_sample*F_sample/1e6
            
        if(verbose):
            plt.figure(figsize=(7, 6))
            plt.grid()
            plt.xlabel("Frequency (MHz)")
            plt.ylabel("Power (dBFS)")
            plt.plot(xf, 10*np.log(X_norm)/np.log(10), label = "real ideal FFT")
            plt.legend()
            plt.show()
            
        return xf, X_norm
    ###############################################################################      
    # returns the windowing function corresponding to data_in as a list
    ###############################################################################  
    def get_window(self, data_in):
        
        #Finds 50% of the number of samples
        N_samples = int(len(data_in)*0.5)
        
        #creates windowing function
        window = list(signal.tukey(N_samples, 0.5))
        
        N_samples_left = int(len(data_in)*0.25)
        zeros_left = [0]*N_samples_left
        
        N_samples_right = int(len(data_in)) - N_samples - N_samples_left
        zeros_right = [0]*N_samples_right
        
        #adds zeros to each end of the windowing function
        window = zeros_left + window + zeros_right
        
        return window
    
    ###############################################################################      
    # gets the normalized fft and frequency given data_in a list. 
    ###############################################################################  
    def get_power(self, data_in, verbose=False):
        
        #gets windowing function corresponding to data_in
        window = self.get_window(data_in)
        
        #gets fft of data_in  by applying windowing function
        xf, x_norm = self.get_chirp_powerspectrum(data_in, self.F_sample, window, verbose)
        
        return xf, x_norm
    
    ###############################################################################  
    # finds the incoherent average of a list of lists data_adrv9009 and averages over N_samples number of times.
    # N_samples is an integer.  Returns x - a frequency list and R_avg list the averaged power in dBFS.
    ###############################################################################  
    def get_average_power(self, data_adrv9009, N_samples):
        R_sum = 0
        window = self.get_window(data_adrv9009[0])
        for i in range(N_samples):
            x, R = self.get_chirp_powerspectrum(data_adrv9009[i], F_sample, window)
            R_sum += R

        R_avg = R_sum/N_samples

        return x, R_avg
        
###############################################################################
# This class generates a quantized chirp, saves the chirp as text_name, and plots the FFT 
# It takes as input F_sample = the sample rate of the tranceiver, 
# N_sample is the number of samples which defaults to the size of the bram 40 bits*4096,
# F_center is the center frequency of the chirp, B_chirp is the chirp bandwidth, T_chirp
# is the chirp time duration, and text_name is where the txbram_data is saved to.
# The text file text_name should then be scp'd to the zcu102.
###############################################################################
class generate_ideal_chirp(chirp_wave):
    
    F_sample = 0
    F_center = 0
    B_chirp = 0
    T_chirp = 0
    K_chirp = 0
    T_zeros = 0
    bits = 1
    
    ###############################################################################
    # constructor takes args and writes to class vars.
    ###############################################################################
    def __init__(self, F_sample = 245.76e6, F_center = 0, B_chirp = 30e6, T_zeros = 55e-6, T_chirp = 20e-6, bits = 1):
        
        self.F_sample = F_sample
        self.F_center = F_center
        self.B_chirp = B_chirp
        self.T_chirp = T_chirp 
        self.T_zeros = T_zeros
        self.bits = bits
        self.K_chirp = B_chirp/T_chirp

    ###############################################################################
    # Takes the args from constructor and creates a quantized and plots the chirp.  It
    # returns xf (frequency) and R (power) of FFT plot.
    ###############################################################################
    def make_ideal_chirp(self, verbose = False):

        data_i, data_q, window = self.make_waveform(verbose = verbose)
        
        mychirp_spect = chirp_spectrum_analysis(F_sample = self.F_sample, F_center = self.F_center, B_chirp = self.B_chirp, T_chirp = self.T_chirp_act)
        xf, X_norm = mychirp_spect.get_chirp_powerspectrum(data = data_i+1j*data_q, F_sample = self.F_sample, window = window, verbose= verbose)
        
        return xf, X_norm, data_i+1j*data_q
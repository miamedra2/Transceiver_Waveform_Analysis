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
    
    ###############################################################################
    # Constructor of Squre_Wave sets the global fields
    ###############################################################################
    def __init__(self, F_in = 10e6, F_sample = 245.76e6, bits = 16):
        
        self.F_in = F_in
        self.F_sample = F_sample
        self.bits = bits

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
            #plt.xlim(0, int(sample_rate/2)/1e6)
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
    N = 0
    F_actual = 0
    channel_spacing = 0
    M = 0

    ###############################################################################
    # Constructor of Squre_Wave sets the global fields
    ###############################################################################
    def __init__(self, F_in = 10e6, F_sample = 245.76e6, N = 256000):
        self.F_sample = F_sample
        self.F_in = F_in
        self.N = N
        #self.F_actual = self.__get_fin(F_in, F_sample, N)
    
    ###############################################################################
    # Makes a quantized square wave corresponding to F_in - the desired frequency
    ###############################################################################
    def __get_onebitsine(self, F_in, F_sample):
        squarewave_i = []
        squarewave_q = []
    
        N  = self.N
        N_2pi = F_sample/F_in
        endval = int(N/N_2pi)*N_2pi
        amplitude = 32767
    
        sample_index = np.arange(0, endval)
    
        data_i = np.cos((2*np.pi*sample_index)/N_2pi)
        data_q = np.sin((2*np.pi*sample_index)/N_2pi)
    
        squarewave_i = [amplitude*2*(np.round(elem)-.5) for elem in 0.5*(data_i+1)]
        squarewave_q = [amplitude*2*(np.round(elem)-.5) for elem in 0.5*(data_q+1)]

        return np.array(squarewave_i), np.array(squarewave_q)
                
    ###############################################################################
    # Finds a frequency close to F_appox that will select all the ADC codes- based on 
    # the sample rate and the number of data points.
    ###############################################################################
    def __get_fin(self, F_in, F_sample, N):
        channel_spacing = F_sample/N
        M = 2*round(F_in/(2*channel_spacing))+1
        F_out = M/N*F_sample
        
        print("M: " + str(M) +" N:" + str(N) + " Channel Spacing: " + str(channel_spacing) + " F_actual: " + str(F_out))
        
        self.channel_spacing = channel_spacing
        self.M = M
    
        return F_out
    ###############################################################################
    # gets a quantized sine data based on F_actual.  Also plots if plot_en is True
    ###############################################################################
    def __make_waveform(self, plot_en = True):  
        
        F_in = self.F_in
        #F_actual = self.F_actual
        F_sample = self.F_sample
        
        #data_i, data_j =  self.__get_onebitsine(F_actual, F_sample)  
        data_i, data_j =  self.__get_onebitsine(F_in, F_sample)  
    
        if(plot_en==True):
            plt.figure(figsize = (6,6))
            plt.xlim(0,30)
            plt.plot(data_i, label = "Ideal Data")
            plt.xlabel('Time (Samples)')
            plt.ylabel('Amplitude (ADC Code)')
            plt.legend()
            plt.title("Ideal Waveform")
            plt.show()
    
        return data_i
    
    ###############################################################################
    # makes a waveform corresponding to F_actual, F_sample, and N_sample
    ###############################################################################
    def make_ideal_waveform(self, verbose = False):
        data_i = self.__make_waveform(plot_en = verbose)
        
        spur_analysis = Quantization_Spur_Analysis(F_in = self.F_in, F_sample = self.F_sample, bits = 1)
        xf, R_dB, w_cg, w_ig = spur_analysis.getFFT_noise(data = data_i, N_t = 1024*5, plot_en = verbose)
        
        return xf, R_dB, data_i
        
###############################################################################
# Square_Wave class builds on GenerateSineWaves to put the output of a quantized squarewave
# into a file suitable for the txbram in the zcu102.  F_in is the input frequency, F_sample
# is the sample rate of the transceiver, N is the number of samples to generate, and
# text_name is the name of file where the txbram data is saved to.
###############################################################################
class Square_Wave(GenerateSinewaves):
    
    text_name = None
    
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
    def __init__(self, F_in = 10e6, F_sample = 40*153.6e6, N = 4096*40, text_name = "tx_bram.txt"):
        
        GenerateSinewaves.__init__(self, F_in = F_in, F_sample = F_sample)
        self.F_sample = F_sample
        self.N = N
        self.F_in = F_in
        self.text_name = text_name
    
    ###############################################################################
    # Takes in data - a numpy array - and puts the data in 40 bit chunks in the text_name
    # file to be used for the loading data into the txbram on the zcu102. The input data is
    # a numpy array consisting of a square wave whose amplitude ranges from [-32767, 32767].  
    ###############################################################################        
    def make_txbram_pattern(self, data):
        
        amplitude = 32767
        
        # quantize data to a list of 0s and 1s
        data = [int(0.5*(np.real(elem) + amplitude)/amplitude) for elem in data]
        
        # create tx_bram_list to put in the 40 bit chunks of data
        tx_bram_list = []
        
        # loop that increments by 80.  Since devmem write is constrained to 64 bit chunks, 
        # I append 64 bit chunk into the list then the remainaing 16 bit chunk of data.  
        # Then put.sh puts the first chunk at eg 0x80010000 increments the address by 64 bits (8bytes) 
        # and puts the remaining 16 bits at 0x8001000C.
        for i in range(0,len(data), 80):
             
            #bitshift data by index j and collect in pattern
            pattern = 0
            for j in range(64):
                try:
                    pattern = (data[i+j] << j) | pattern
                except:
                    break
        
            # append pattern to the list
            tx_bram_list.append(pattern)
            
            #bitshift data by index j and collect in pattern
            pattern = 0
            for j in range(16):
                try:
                    pattern = (data[i+j+64] << j) | pattern
                except:
                    break
        
            # append pattern to the list
            tx_bram_list.append(pattern)
            
        #puts the list into a csv file with name self.text_name
        pd.DataFrame(tx_bram_list).to_csv(self.text_name, index=False, header=False)
        
    ###############################################################################
    # main method in this class.  Makes an ideal square wave given the params in the 
    # global vars and parses the output waveform into 40 bit chunks for the txbram.  
    # Puts the 40 bit chunks into text_name. 
    ###############################################################################  
    def make_waveform(self, verbose = True):
        
        xf, R_dB, data = self.make_ideal_waveform(verbose = verbose)
        
        self.make_txbram_pattern(data)
        
        return xf, R_dB, data
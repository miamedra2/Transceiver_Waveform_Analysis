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
# simple_chirp contains a few methods useful in creating arbitrary waveforms for loading into 
# txbram on the zcu102/adrv9009 hdl design.  Contains three methods, make_edge(), find_offset(),
# and make_waveforem.
###############################################################################
class simple_chirp:
    
    ###########################################################################
    # Given an integer input_data returns data of the form 4'b1100 or 6'b111000 where input data
    # in the first case is 2 and in the second case 3.  
    ###########################################################################
    def make_edge(self, input_data, verbose=False):
    
        data = 0
    
        for i in range(input_data):
            data = ( 1 << (i+input_data) ) | data 
            if(verbose == True):
                print("1", end = " ")
        
        for i in range(input_data):
            if(verbose == True):
                print("0", end = " ")
        
        return data

    ###########################################################################
    # Given an integer input_data returns the maximum bit index- defined as the max non-zero value.  
    # For example, if input_value is 0x18 = 8'b00011000, the returned value is 5 since that is 
    # the maximum non-zero value. 
    ###########################################################################
    def find_offset(self, input_value):
    
        index = 0
        data_temp = input_value
    
        while(data_temp != 0):
            index += 1
            data_temp = data_temp >> 1
        
        return index

    ###########################################################################
    # this function creates an example waveform that can be loaded to the txbram.  It creates a waveform
    # that looks like:
    # 10
    # 1100
    # 111000
    # And parses the data such that it has the form 101100111000 and bins into 40 bit chunks in simple_chirp.txt.
    # simple_chirp.txt could then be put onto the zcu102 using scp and loaded into tx_bram memory region - 0x80010000.
    ###########################################################################
    def make_waveform(self, input_data, verbose = False):
        val = 0
        data_array = []
        input_data = input_data+1
    
        #put make_edge in data_array
        for i in reversed(range(1,input_data)):
            data_array.append(self.make_edge(i, verbose))
    
        print()
        print("elems of Data Array")
        if(verbose == True):
            print([hex(elem) for elem in data_array])
        
        #convert array values to single integer(in python val can be arbitrarily large)
        val = data_array[0]
        bitshift = 0
    
        for i in range(1, len(data_array)):
        
            bitshift = self.find_offset(val)
            if(verbose == True):
                print(hex(val), bitshift)
            val  = (data_array[i] << bitshift) | val
        
        tx_bram_list = [] 
    
        if(verbose):
            print(hex(val))
    
        #bitshift val by 40 to create list of 40 bit elems
        while(val != 0):
            tx_bram_list.append(val & 0xffffffffff)
            val = val >> 40
    
        pd.DataFrame(tx_bram_list).to_csv("simple_chirp.txt", index=False, header=False)
        
        if(verbose):
            print(tx_bram_list)
    
        return tx_bram_list
        
###############################################################################
# Square_wave redefines make_waveform in simple_chirp to create a square wave
# pattern.
###############################################################################
class square_wave(simple_chirp):
    
    def make_waveform(self, divide_by_input, pattern_size = 8191, verbose=False):
        
        TX_BRAM_MAX = 8191  
        
        #print out frequency to the user
        print("freq: ", 153.6*40/divide_by_input)
        
        #divide factor by two to be in units of make_edge
        divide_by_input = int(divide_by_input/2)
        
        #get the pattern of the value to write to the bram 
        pattern = self.make_edge(divide_by_input)
        temp_val = pattern
        
        #find how many bits the pattern takes in memory
        offset_val = self.find_offset(pattern) 
        
        if(verbose):
            print("Value:", hex(pattern), "Size:", offset_val, "bits")

        #find the number of times to apply the pattern to the 8191 txbram buffer
        N = int(pattern_size*40/offset_val)
        
        #duplicate the pattern N-1 times
        for i in range(N-1):
            pattern = (pattern << offset_val) | pattern
        
        #initialize list to be put into the tx_bram
        tx_bram_list = []
        
        #iterate through max size of 
        for i in range(TX_BRAM_MAX):
            if((pattern != 0) & (i < pattern_size)):
                tx_bram_list.append(pattern & 0xFFFFFFFFFF)
                pattern = pattern >> 40
            else:
                tx_bram_list.append(0)
        
        pd.DataFrame(tx_bram_list).to_csv("tx_bram.txt", index=False, header=False)
        
        return tx_bram_list

###############################################################################
# Square_wave redefines make_waveform in simple_chirp to create a chirp wave
# pattern.  It takes as input F_sample = the sample rate of the tranceiver, 
# N_sample is the number of samples which defaults to the size of the bram 40 bits*4096,
# F_center is the center frequency of the chirp, B_chirp is the chirp bandwidth, T_chirp
# is the chirp time duration, and text_name is where the txbram_data is saved to.
###############################################################################
class chirp_wave(simple_chirp):
    
    F_sample = 0
    N_sample = 0
    F_center = 0
    B_chirp = 0
    T_chirp = 0
    K_chirp = 0
    text_name = "tx_bram.txt"
    
    ###############################################################################
    #Constructor of chirp_wave sets the global fields
    ###############################################################################
    def __init__(self, F_sample = 153.6*40e6, N_sample = 4096*40, F_center = 700e6, B_chirp = 30e6, T_chirp = 20e-6, text_name="tx_bram.txt"):
        
        self.F_sample = F_sample
        self.N_sample = N_sample
        self.F_center = F_center
        self.B_chirp = B_chirp
        self.T_chirp = T_chirp 
        self.K_chirp = B_chirp/T_chirp
        self.text_name = text_name
    
    ###############################################################################
    # Given a list t_n, and chirp params K_chirp, and F_centers returns an array containing an unquantized chirp
    ###############################################################################
    def chirp_exp(self, t_n, K_chirp, F_center):
        return np.exp(1j*2*np.pi*(t_n*F_center+1/2*K_chirp*t_n**2))
    
    ###############################################################################
    # Redefines make_waveform in simple_chirp to create a chirp based on the constructor arguments of chirp_wave.  
    # Saves the chirp_waveform by putting 40 bit chunks into the file specified by text_name.
    ###############################################################################
    def __make_waveform(self, plot_en=False):
        
        TX_BRAM_MAX = 8191
        
        F_sample = self.F_sample
        N_sample = self.N_sample
        F_center = self.F_center
        B_chirp = self.B_chirp
        T_chirp = self.T_chirp
        K_chirp = self.K_chirp
        
        #creates a np array centered on 0 from -N_sample/2 to N_sample_2
        n = np.arange(-N_sample/2,N_sample/2)
        
        #divides the array by the sample rate to put in units of time
        t = n/F_sample
        
        #gets the chirp and makes an array in terms of x
        x = np.array([self.chirp_exp(elem[1], K_chirp, F_center) if abs(elem[1])<=T_chirp/2 else 0 for elem in enumerate(t)])
        
        #plots the chirp
        if(plot_en == True):
            plt.plot(np.real(x))
            plt.show()
        
        #normalizes the chirp to 1 and then rounds (0,1) and casts each elem to an int to get a list   
        quant_data = [int(elem) for elem in np.round(0.5*(np.real(x)+1))]
        
        return quant_data
    
    def make_ideal_chirp(self, verbose = True):
        
        quant_data = self.__make_waveform(plot_en = verbose)
        
        chirp_analysis = chirp_spectrum_analysis(F_sample = self.F_sample, F_center = self.F_center, B_chirp = self.B_chirp, T_chirp = self.T_chirp)
        xf, R_dB = chirp_analysis.get_power(data_in = quant_data, verbose = True)
        
        return xf, R_dB, quant_data
        
            
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
            plt.plot(data)
            plt.plot(window)
            plt.show()
            
        #gets N_sample from the length of data
        N_sample = len(data)

        #get the fft of the product
        X = np.fft.fftshift(scipy.fftpack.fft(np.fft.fftshift(product))) 
        
        #normalize the fft to 2**16/2-1
        X_norm = (abs(X)*np.sqrt(K_chirp)/(F_sample*32767))**2

        #create frequency range according to the number of samples and sample rate
        xf = np.arange(-int(N_sample/2), int(N_sample/2))/N_sample*F_sample/1e6

        if(verbose):
            plt.figure(figsize=(7, 6))
            plt.grid()
            plt.xlabel("Frequency (MHz)")
            plt.ylabel("Power (dBFS)")
            plt.xlim(0,6000)
            plt.plot(xf, 10*np.log(X_norm)/np.log(10))
            plt.show()
            
        return xf, X_norm
    ###############################################################################      
    # returns the windowing function corresponding to data_in as a list
    ###############################################################################  
    def get_window(self, data_in):
        
        #Finds 80% of the number of samples
        N_samples = int(len(data_in)*0.8)
        
        #creates windowing function
        window = list(signal.tukey(N_samples, 0.5))
        
        #Finds 10% of the number of samples
        zeros = [0]*int(len(data_in)*.1)
        
        #adds zeros to each end of the windowing function
        window = zeros + window + zeros
        
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
    N_sample = 0
    F_center = 0
    B_chirp = 0
    T_chirp = 0
    K_chirp = 0
    
    ###############################################################################
    # constructor takes args and writes to class vars.
    ###############################################################################
    def __init__(self, F_sample = 153.6*40e6, N_sample = 4096*40, F_center = 700e6, B_chirp = 30e6, T_chirp = 20e-6, text_name="tx_bram.txt"):
        
        self.F_sample = F_sample
        self.N_sample = N_sample
        self.F_center = F_center
        self.B_chirp = B_chirp
        self.T_chirp = T_chirp 
        self.K_chirp = B_chirp/T_chirp
        self.text_name = text_name
        
    ###############################################################################
    # Takes in data - a numpy array - and puts the data in 40 bit chunks in the text_name
    # file to be used for the loading data into the txbram on the zcu102. The input data is
    # a numpy array consisting of a square wave whose amplitude ranges from [-32767, 32767].  
    ###############################################################################        
    def make_txbram_pattern(self, data):
        
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
    # Takes the args from constructor and creates a quantized and plots the chirp.  It
    # returns xf (frequency) and R (power) of FFT plot.
    ###############################################################################
    def make_waveform(self, verbose = True):
    
        xf, R_dB, data = self.make_ideal_chirp(verbose = verbose)
        
        self.make_txbram_pattern(data)
        
        return xf, R_dB, data
    
###############################################################################
# ADRV9009_Data deals is a parent class that helps to get a handle, Write/Read to the ADRV9009 DAC/ADC
# or just read data from the ADC.  Input params are the ip (note the string) and the rx_buffer_size.
# Other params relevent to the ADRV9009 (eg. lo_trx, calibration, etc) can be set by accessing
# variables in sdr.  Doing a sdr.__dir__ should expose all the methods vars available.
###############################################################################
class ADRV9009_Data:
    
    sdr = 0
    rx_buffer_size = 1024
    
    ###############################################################################
    # constructor takes args and writes to class vars.
    ###############################################################################    
    def __init__(self, ip = "ip:192.168.1.21", rx_buffer_size = 1024*32):
        
        self.get_handle_ADRV9009(ip)
        self.rx_buffer_size = rx_buffer_size
    ###############################################################################        
    #gets a handle given the ip.  Notice the ip string it doesn't just consist of the ip address
    ###############################################################################
    def get_handle_ADRV9009(self, ip = "ip:192.168.1.21"):
        ADRV9009_Data.sdr = adi.adrv9009(uri=ip)  
        
    ###############################################################################    
    #gets ADC data given input waveforms data_i/q.  These waveforms should be
    #90 degrees out of phase for the quad mixer
    ###############################################################################
    def get_ADC_data_RW(self, data_i, data_q):

        sdr = ADRV9009_Data.sdr
        
        sdr.tx_destroy_buffer()
        sdr.rx_destroy_buffer()
        sdr.rx_buffer_size = self.rx_buffer_size
        sdr.tx_enabled_channels = [0, 1]
        # Create a sinewave waveform
        #N = 256000
        fs = int(sdr.tx_sample_rate)
        fc = 40000000
        #i0 = np.cos(2 * np.pi * t * fc) * 30000
        #q0 = np.sin(2 * np.pi * t * fc) * 30000 
        iq = data_i + 1j * data_q
        fc = -30000000
        ts = 1 / float(fs)
        N=len(data_i)
        t = np.arange(0, N * ts, ts)
        i1 = np.cos(2 * np.pi * t * fc) * 2
        q1 = np.sin(2 * np.pi * t * fc) * 2
        iq2 = i1 + 1j * q1
        # Send data to both channels
        sdr.tx([iq, iq2])
        data_out = sdr.rx()
        sdr.tx_destroy_buffer()
        sdr.rx_destroy_buffer()
        
        data_out = list(data_out[0])
    
        return data_out
    ###############################################################################    
    # Gets ADC data time series data from the ADRV9009.  The number of samples is set according to rx_buffer_size.
    # These waveforms should be 90 degrees out of phase for the quad mixer.  Parses the data to return the 0th channel
    # as a list
    ###############################################################################
    def get_ADC_data_Read(self):

        sdr = ADRV9009_Data.sdr
        sdr.tx_destroy_buffer()
        sdr.rx_destroy_buffer()
        sdr.rx_buffer_size = self.rx_buffer_size
        
        data_out = sdr.rx()
        sdr.tx_destroy_buffer()
        sdr.rx_destroy_buffer()
    
        data_out = list(data_out[0])
        
        return data_out
        
###############################################################################
# A layer of abstraction used to configure the ADRV9009 data collection params (that could be set
# by just setting sdr.param).  This protects from writing to read only params and making sure,
# the correct type is used when writing.  Also, saves the params as local vars that cans be used
# later to save to a dataframe by doing vars(ADRV9009_Config_1).
# It takes as input f_lo the frequency of the local oscillator, rx_gain the hardware gain
# of the receiver, rx_buffer_size size of number of samples to collect, tx_gain the 
# transmit gain of the ADRV9009 transmit section, calibrate_rx_phase_correction_en - 
# comment on the calibration
###############################################################################
class ADRV9009_Config(ADRV9009_Data):
    
    ip = None
    trx_lo = None
    rx_buffer_size = None
    calibrate_rx_phase_correction_en = None
    calibrate_rx_qec_en = None
    calibrate_tx_qec_en = None
    calibrate = 1
    
    frequency_hopping_mode = None
    frequency_hopping_mode_en = None
    calibrate_rx_phase_correction_en = None
    calibrate_rx_qec_en = None
    calibrate_tx_qec_en = None
    calibrate = None
    gain_control_mode_chan0 = None
    gain_control_mode_chan1 = None
    rx_hardwaregain_chan0 = None
    rx_hardwaregain_chan1 = None
    tx_hardwaregain_chan0 = None
    tx_hardwaregain_chan1 = None
    rx_rf_bandwidth = None
    tx_rf_bandwidth = None
    rx_sample_rate = None
    tx_sample_rate = None
    
    
    ###############################################################################
    # constructor takes args and writes to df_params.
    ############################################################################### 
    def __init__(self, ip = "ip:192.168.1.21", f_lo = 700000000, rx_gain = 25, rx_buffer_size = 1024*32,
                 tx_gain = -14, calibrate_rx_phase_correction_en = 0, calibrate_rx_qec_en = 0, calibrate_tx_qec_en = 0, 
                 calibrate = 1):
        
        ADRV9009_Data.__init__(self, ip = ip, rx_buffer_size = rx_buffer_size)
        
        #get params from the ADRV9009
        self.get_params_from_ADRV9009(ip)
        
        #set ip value in df
        self.ip = ip
        self.trx_lo = int(f_lo)        
        self.rx_hardwaregain_chan0 = rx_gain
        self.tx_hardwaregain_chan0 = tx_gain
        self.calibrate_rx_phase_correction_en = calibrate_rx_phase_correction_en
        self.calibrate_rx_qec_en = calibrate_rx_qec_en
        self.calibrate_tx_qec_en = calibrate_tx_qec_en
        self.calibrate = calibrate
        self.rx_buffer_size = rx_buffer_size
        
        #set the params in the ADRV9009
        self.set_params_ADRV9009()
        
    ###############################################################################    
    #gets the the configuration parameters from the adrv9009 and returns as a dataframe
    ###############################################################################
    def get_params_from_ADRV9009(self, ip = "ip:192.168.1.21"):
        
        #set ip value in df
        self.ip = ip

        #get handle of the adrv9009 from df
        sdr = adi.adrv9009(uri = self.ip)  
        
        #assign to class variable
        ADRV9009_Data.sdr = sdr

        #get params from the adrv9009 and assign them to df
        self.frequency_hopping_mode = sdr.frequency_hopping_mode
        self.frequency_hopping_mode_en = sdr.frequency_hopping_mode_en
        self.calibrate_rx_phase_correction_en = sdr.calibrate_rx_phase_correction_en
        self.calibrate_rx_qec_en = sdr.calibrate_rx_qec_en
        self.calibrate_tx_qec_en = sdr.calibrate_tx_qec_en
        self.calibrate = 1 #read only
        self.gain_control_mode_chan0 = sdr.gain_control_mode_chan0
        self.gain_control_mode_chan1 = sdr.gain_control_mode_chan1
        self.rx_hardwaregain_chan0 = sdr.rx_hardwaregain_chan0
        self.rx_hardwaregain_chan1 = sdr.rx_hardwaregain_chan1
        self.tx_hardwaregain_chan0 = sdr.tx_hardwaregain_chan0
        self.tx_hardwaregain_chan1 = sdr.tx_hardwaregain_chan1
        self.rx_rf_bandwidth = sdr.rx_rf_bandwidth
        self.tx_rf_bandwidth = sdr.tx_rf_bandwidth
        self.rx_sample_rate = sdr.rx_sample_rate
        self.tx_sample_rate = sdr.tx_sample_rate
        self.trx_lo = sdr.trx_lo
        self.rx_buffer_size = sdr.rx_buffer_size
        #df_params['tx_buffer_size'] = sdr.tx_buffer_size

    ###############################################################################    
    #sets the parameters of the adrv9009 based on the df_in - which is a dataframe
    ###############################################################################
    def set_params_ADRV9009(self):
        
        sdr = ADRV9009_Config.sdr
        sdr.tx_destroy_buffer()
        sdr.rx_destroy_buffer()
        sdr.tx_cyclic_buffer = True
        sdr.trx_lo = self.trx_lo
        sdr.gain_control_mode = 'manual'
        sdr.rx_buffer_size = self.rx_buffer_size
        #sdr.tx_buffer_size = int(df_in['tx_buffer_size'][0])
        sdr.rx_hardwaregain_chan0 = self.rx_hardwaregain_chan0
        sdr.tx_hardwaregain_chan0 = self.tx_hardwaregain_chan0
        sdr.calibrate_rx_phase_correction_en = self.calibrate_rx_phase_correction_en
        sdr.calibrate_rx_qec_en = self.calibrate_rx_qec_en
        sdr.calibrate_tx_qec_en = self.calibrate_tx_qec_en
        sdr.calibrate = self.calibrate

        
###############################################################################
# This class is supposed to deal with all the HDF5 table read/write.  The only input
# is the name of the HDF5 file to be created
###############################################################################
class Collect_HDF5:
    
    hdf5_datafile = None

    ###############################################################################
    # constructor takes args and writes to df_params.
    ############################################################################### 
    def __init__(self, hdf5_datafile = 'adrv9009.h5'):
        
        self.hdf5_datafile = hdf5_datafile

    ###############################################################################
    # Used to put the variables in an object instantantiated from some class into a pandas dataframe.
    # Input is the object itself and the output is the dataframe.  This is used to record settings
    # like the tranceiver params and the ADRV9009 Data Configuration Params and store them in an
    # hdf5 table.
    ############################################################################### 
    def make_df_from_object_params(self, obj_input):
    
        #makes a dataframe with columns param and value for all the global variables in the object
        df = pd.DataFrame(vars(obj_input).items(), columns = ["Param", "Value"])

        #transposes the dataframe and makes the elems in Param the column 
        df_out = pd.DataFrame()
        df_out = df_out.append(df['Value'].to_list()).T
        df_out.columns = df['Param'].to_list()
        
        #dataframes default to object type, so this looks at the first row of each column and sets the type according
        #to the value their.  
        for elem in df_out:
            elem_type = type(df_out[elem][0])
            if(elem_type == str):
                elem_type = '|S'
            elif(elem_type == int):
                elem_type = np.int64
    
            #apply the type of the zeroth value of elem to the column elem
            df_out[elem] = df_out[elem].astype(str).astype(elem_type)
    
        return df_out
        
    ###############################################################################
    # Returns a list of keys corresponding to self.hdf5_datafile
    ###############################################################################     
    def get_keys_hdf5(self):
        #get the class variable
        hdf5_datafile = self.hdf5_datafile
        
        #opens the hdf5 file
        hdf5_store = pd.HDFStore(hdf5_datafile)
        
        #gets list of keys
        key_list = [key for key in hdf5_store.keys()]
        
        #closes the file
        hdf5.close()
        
        return key_list

    ###############################################################################
    # Saves a df_in a dataframe to self.hdf5_datafile as a table with name given by title
    ############################################################################### 
    def save_df_to_hdf5(self, df_in, hdf5_datafile, title):
        #opens the hdf5 file
        hdf5_store = pd.HDFStore(hdf5_datafile)

        #puts df_in into a table called config_params
        hdf5_store.put(title, df_in)
        
        hdf5_store.close()
        
        print("saved dataframe to " + str(title) + " table")
        
    ###############################################################################        
    #gets the parameters set in the hdf5 file
    ###############################################################################
    def get_table_from_hdf5(self, params = 'config_params'):
         #opens the hdf5 file
        
        hdf5_datafile = self.hdf5_datafile
            
        hdf5_store = pd.HDFStore(hdf5_datafile)

        #gets the table from the hdf5 file and sets it to a dataframe to be returned
        df_out = hdf5_store[params]

        #closes the hdf5 file
        hdf5_store.close()

        return df_out
    
        def save_array_data_to_hdf5(self, nparray_in, hdf5_datafile, title):
        
            handle = h5py.File(hdf5_datafile, 'a')
        try: 
            #try to delet the array in the hdf5 file
            del handle[title]
            handle[title].value
        except:
            #delete failed ignore the warning that delete failed
            pass
        finally:
            #append the array
            handle.create_dataset(title, data = nparray_in)
            handle.close()
            
    ###############################################################################
    # Converts complex128 to a dtype consisting of 16 bit real and imag terms.  Input is a 
    # pandas dataframe and the output is a numpy array.  This function basically loops through
    # the pandas dataframe and gets the real/imag parts and puts them in the output array x.
    # There might be a better pythonic way to do this because it's time consuming..
    ###############################################################################             
    def complex128_to_complexint(self, df_in):
    
        #complex baseband representation
        dtype = np.dtype([('re', np.int16), ('im', np.int16)])
        
        #make array of N x M dimensional array from df_in
        x = np.zeros((df_in.shape[1], df_in.shape[0]), dtype)
    
        #loop through the df and get the real/imag parts
        for m in range(df_in.shape[1]):
            for n in range(df_in.shape[0]):
                x[m,n]['re'] = np.real(df_in[m][n])
                x[m,n]['im'] = np.imag(df_in[m][n])  
                
        #transpose so that x has the right form.
        x = np.transpose(x)
        
        return x

    ###############################################################################
    # Saves a numpy array to a an hdf5_datafile.  Input is the nparray_in which is the 
    # data to be saved and title - a string that gives a name to the saved data.
    ###############################################################################   
    def save_array_data_to_hdf5(self, nparray_in = None, title = "data"):
        
        if(nparray_in == None):
            print("No input array. Saving to hdf5 failed")
            return
        
        handle = h5py.File(self.hdf5_datafile, 'a')
        try: 
            #try to delet the array in the hdf5 file
            del handle[title]
            handle[title].value
        except:
            #delete failed ignore the warning that delete failed
            pass
        finally:
            #append the array
            handle.create_dataset(title, data = nparray_in)
            handle.close()
    
    ###############################################################################
    # Gets a complexint table from the .hdf5 file by name params.  This function will fail if
    # the table with name params does not have data type 
    # dtype = np.dtype([('re', np.int16), ('im', np.int16)]) -  corresponding to 
    # complex128_to_complexint()
    ############################################################################### 
    def get_complexint_data(self, params='raw_data', hdf5_datafile = 'data.hdf5'):
        
        #gets the data array corresponding to title
        hf = h5py.File(hdf5_datafile, 'r')
        n = hf.get(params)
        n1 = np.array(n)
        hf.close()
    
        #data needs to be converted from dtype to complex128 for math libraries- time consuming
        n2 = np.zeros(n1.shape, np.complex128)
        for m in range(n1.shape[0]):
            for n in range(n1.shape[1]):
                n2[m,n]= n1[m][n]['re'] + 1j*n1[m][n]['im'] 
            
        #now convert back to pandas dataframe to be consistent with everything else in the chirp class
        df = pd.DataFrame(n2, columns=[elem for elem in range(n2.shape[1])])
    
        return df
            
################################################################################################
# The purpose of this class is to extract a single chirp from some time series data.  It takes in 
# the sample frequency of the ADRV9009 receiver, F_center - the center frequency of the chirp,
# B_chirp the bandwidth of the chirp, T_chirp - the time duration of the chirp, and F_LO the LO
# frequency of the ADRV9009 receiver.  If these input params don't reflect the member vars in 
# generate_ideal_chirp, get_onechirp won't work.  
################################################################################################
class Trigger_ChirpData(chirp_spectrum_analysis, chirp_wave):
    
    F_sample = 0
    F_center = 0
    B_chirp = 0
    T_chirp = 0
    K_chirp = 0
    F_LO = 0
    
    
    ###############################################################################
    # constructor takes args and writes to class vars.
    ###############################################################################
    def __init__(self, F_sample = 245.76e6, F_center = 700e6, B_chirp = 30e6, T_chirp = 20e-6, F_LO = 700e6):
        
        self.F_sample = F_sample
        self.F_center = F_center
        self.B_chirp = B_chirp
        self.T_chirp = T_chirp 
        self.K_chirp = B_chirp/T_chirp
        self.F_LO = F_LO
        
    ###############################################################################
    # returns the global params 
    ###############################################################################
    def get_chirp_params(self):
        
        return self.F_center, self.B_chirp, self.F_sample, self.T_chirp, self.K_chirp, self.F_LO

    ###############################################################################
    # Returns a normalized FFT.  Input is data- a list of time series data, sample_rate- the
    # sample rate of the ADRV9009 receiver, B is the number of bits (1-16), and plot en
    # plots the fft.  
    ###############################################################################
    def getchirpFFT(self, data, sample_rate, B, plot_en):
        N_samples = len(data)

        X = np.fft.fftshift(scipy.fftpack.fft(data))
        X_dBFS = 20*(np.log(abs(X)/(2**(B-1)*N_samples))/np.log(10))

        xf = np.arange(-int(N_samples/2), int(N_samples/2))/N_samples*sample_rate/1e6

        if(plot_en):
            self. makefftplot(xf, X_dBFS)

        return xf, X_dBFS
    
    ###############################################################################
    # From plot_chirp_response.m.  Used to extract a single chirp from time-series data
    # consisting of many chirps.  Issues that may arise from the script is that y[m+n]
    # may on occasion go over bounds.  This can be adjusted by setting N_sample- the number
    # of samples to collect corresponding to the isolated chirp.  Default for this is
    # the Number of sample corresponding to twice the chirp time.
    ###############################################################################
    def get_onechirp(self, data_i, data_q, verbose):
        
        #Get the chirp parameters- make sure these are set correctly
        F_center, B_chirp, F_sample, T_chirp, K_chirp, F_LO = self.get_chirp_params()
        
        #Want the offset of the chirp with respect to the ADRV9009 receiver
        F_center = F_center - self.F_LO

        #Choose number of samples equal to twice the chirp duration
        N_sample = int(2*T_chirp*F_sample)
        self.N_sample = N_sample
        
        #Create a fourier/time array
        n = np.arange(-N_sample/2,N_sample/2)
        t = n/F_sample

        #create an ideal waveform for comparison
        m = np.array([elem[0] for elem in enumerate(t) if abs(elem[1])<=T_chirp/2])
        x = np.array([self.chirp_exp(elem[1], K_chirp, F_center) if abs(elem[1])<=T_chirp/2 else 0 for elem in enumerate(t)])

        # combine the real and imag data
        y = pd.Series(data_i)+1j*pd.Series(data_q)
        M_sample = len(y)

        #get frequency array
        f = np.arange(-M_sample/2,M_sample/2)/M_sample*F_sample
        
        #get max amplitude
        A = max(abs(y))
        
        #pulse compression
        Z = np.fft.fftshift(scipy.fftpack.fft(np.fft.fftshift(y)))*np.exp(1j*np.pi*(f-F_center)**2/K_chirp)/np.sqrt(1j)
        z = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(Z)))

        #find the upper/lower bounds that correspond to 2N amount of data 
        lower_bound = int(np.round(T_chirp*F_sample)) 
        upper_bound = int(np.round(M_sample - T_chirp*F_sample)) 
        
        #find the peak of pulse compressed z in order to extract best chirp from raw data y
        m = abs(z[lower_bound:upper_bound]).argmax(axis=0) + lower_bound
        
        if(verbose==True):
            plt.figure(figsize = (6,6))
            plt.title("Raw Data")
            plt.plot(np.arange(0,M_sample)/F_sample*1e6,np.real(y), label = "real")
            plt.plot(np.arange(0,M_sample)/F_sample*1e6,np.imag(y), label = "imag")
            plt.xlabel("time (us)")
            plt.ylabel("Amplitude (ADC Code)")
            plt.legend()
            plt.show()

        if(verbose == True):
            plt.figure(figsize = (6,6))
            plt.title("Extracted Real Data from peak of pulse compressed data")
            plt.plot((t+m/F_sample)*1e6,0.95*A*np.real(x), label = "real ideal")
            z_max = max(abs(z))
            plt.plot(np.arange(0,M_sample)/F_sample*1e6, abs(z)/z_max*20000, label = "pulse compressed")
            plt.plot(np.arange(0,M_sample)/F_sample*1e6, np.real(y), label = "real raw")
            plt.plot((t+m/F_sample)*1e6, 0.9*np.real(y[m+n]), label = "extracted real")
            plt.xlabel("time (us)")
            plt.ylabel("Amplitude (ADC Code)")
            plt.legend()
            plt.show()  

        #extract the n sample chirp
        z = y[m+n]
        
        #scale factor
        A = np.std(z)/np.std(x)

        if(verbose == True):
            plt.figure(figsize = (6,6))
            plt.title("Ideal and extracted data Comparison (Real)")
            plt.plot(t*1e6, A*np.real(x), label = "ideal real")
            plt.plot(t*1e6, np.real(z), label = "extracted real")
            plt.xlabel("Time (us)")
            plt.ylabel("Real Amplitude (ADC Code)")
            plt.legend()
            plt.show()

        if(verbose == True):
            plt.figure(figsize = (6,6))
            plt.title("Ideal and extracted data Comparison (Imag)")
            plt.plot(t*1e6, A*np.imag(x), label = "ideal imag")
            plt.plot(t*1e6, np.imag(z), label = "extracted imag")
            plt.xlabel("Time (us)")
            plt.ylabel("Imag Amplitude (ADC Code)")
            plt.legend()
            plt.show()

        #pulse compress
        f = np.arange(-N_sample/2,N_sample/2)/N_sample*F_sample
        Z = np.fft.fftshift(scipy.fftpack.fft(np.fft.fftshift(z)))
        C = np.exp(1j*np.pi*f**2/K_chirp)/np.sqrt(1j)

        #get an FFT of the isolated z data
        xf, R_out = self.getchirpFFT(np.real(z)+1j*np.imag(z), F_sample, 16, False)


        return np.real(z), np.imag(z), xf, R_out

################################################################################################
# This class is meant to be the interface to the ADRV9009.  It's a child class of ADRV9009_Config, ADRV9009_Data, and Collect_HDF5 
# to collect data based on the params in __init__.  These params are put into a dataframe df_params and saved to hdf5_datafile.  
# Don't redefine the init variables (eg. f_lo, ip, rx_buffer_size) as for now, these new writes won't be written to the ADRV9009.  
# Instead, instantiate another ADRV9009 object or re instantiate and change the arguments.
################################################################################################

class ADRV9009(ADRV9009_Config, Collect_HDF5):
    
    #ADRV9009 Constructor
    def __init__(self, hdf5_datafile = 'adrv9009.h5', ip = "ip:192.168.1.21", rx_buffer_size = 1024*32, f_lo = 700000000, rx_gain = 25, 
                 tx_gain = -14, calibrate_rx_phase_correction_en = 0, calibrate_rx_qec_en = 0, calibrate_tx_qec_en = 0, 
                 calibrate = 1): 
    
        #ADRV9009_Config Constructor 
        ADRV9009_Config.__init__(self, ip = ip, f_lo = f_lo, rx_gain = rx_gain, tx_gain = tx_gain, calibrate_rx_phase_correction_en = calibrate_rx_phase_correction_en, 
                                            calibrate_rx_qec_en = calibrate_rx_qec_en, calibrate_tx_qec_en = calibrate_tx_qec_en, 
                                            calibrate = calibrate)
        
        #HDF5 Constructor
        Collect_HDF5.__init__(self, hdf5_datafile)
        
        #saving df_params from ADRV9009_Config using Collect_HDFf5
        self.save_to_hdf5(self.df_params, self.hdf5_datafile, 'chirp_params')
    
    ###############################################################################    
    # Gets ADC data time series data from the ADRV9009.  The number of samples is set according to rx_buffer_size.
    # These waveforms should be 90 degrees out of phase for the quad mixer.  Parses the data to return the 0th channel
    # as a list
    ###############################################################################
    def get_ADC_data_Read(self):

        sdr = ADRV9009_Data.sdr
        sdr.tx_destroy_buffer()
        sdr.rx_destroy_buffer()
        sdr.rx_buffer_size = self.rx_buffer_size
        
        data_out = sdr.rx()
        sdr.tx_destroy_buffer()
        sdr.rx_destroy_buffer()
    
        data_out = list(data_out[0])
        
        return data_out
    
    
    ###############################################################################
    # Uses the the ADRV9009_Config() and the Trigger_ChirpData() classes together to pull
    # N amount of time series data from the ADRV9009 Receiver.  Output is data_list - list 
    # of lists where each list represents a N_samples of data.  Input is N- the length of the
    # amount of data samples to pull, adrv9009_config/chirp_trigger - the vars corresponding to the 
    # instantiated ADRV9009_Config()/Trigger_ChirpData().
    ###############################################################################
    def get_N_TriggeredData(N = 10, adrv9009_config = None, chirp_trigger = None, verbose = False):
        data_list = []
    
        if(adrv9009_config == None or chirp_trigger == None):
            printf("adrv9009_config or chirp_trigger invalid")

        # Uses a try block because there are instances in which get_onechirp does not succeed in obtaining 
        # N_sample = int(2*T_chirp*F_sample)
        # since the chirp packet might be very near the end of the time series data.  In those instances,
        # I just print failed and try to pull data again.
        while(len(data_list) < N):
            try:
                data = adrv9009_config.get_ADC_data_Read()
                z_real, z_imag, xf, R_out = chirp_trigger.get_onechirp(np.real(data), np.imag(data), verbose)
                data_list.append(z_real + 1j*z_imag)
            except: 
                print("retry")
            
        return data_list

################################################################################################
# The purpose of this class is to average and solve the time jitter problem that occurs when plots from
# plots obtained from Trigger_ChirpData may be offset by a few samples.  Chirp Average fixes the jitter
# problem and then does an incoherent average of the fixed jitter time series data.
################################################################################################
class Chirp_average(Trigger_ChirpData):
    
    F_Cent_rx  = 0
    
    ###############################################################################
    # constructor takes args and writes to class vars.
    ###############################################################################
    def __init__(self, F_sample = 245.76e6, F_center = 700e6, B_chirp = 30e6, T_chirp = 20e-6, F_LO = 700e6):
        
        self.F_sample = F_sample
        self.F_center = F_center
        self.B_chirp = B_chirp
        self.T_chirp = T_chirp 
        self.K_chirp = B_chirp/T_chirp
        self.F_LO = F_LO
        self.F_Cent_rx = F_center - F_LO
        
    ###############################################################################
    # returns the global params 
    ###############################################################################
    def get_chirp_params(self):
        
        return self.F_center, self.B_chirp, self.F_sample, self.N_sample, self.T_chirp, self.K_chirp, self.F_LO, self.F_Cent_rx
    
    ###############################################################################
    # constructor takes args and writes to class vars.
    ###############################################################################
    def get_tau(self, f, Y, plot_en):    
        
        F_center, B_chirp, F_sample, N_sample, T_chirp, K_chirp, F_LO, F_Cent_rx = self.get_chirp_params()
        
        param = self.fit_chirp(f, Y, plot_en)
        tau = 1/F_sample*1e6*param[0]

        return tau, param[1]
        
    ###############################################################################
    # Fits the FFT chirp with a line between -B_chirp/2, B_chirp/2 with a line.  Here f is a 
    # frequency array, Y is the FFT power, B_chirp is the bandwidth of the chirp and F_center
    # is the chirp's center frequency with respect to the LO's receiver.  Eg. F_Center = F_center - F_LO.
    ###############################################################################
    def fit_chirp(self, f, Y, plot_en):
        
        #gets the global params from the class
        F_center, B_chirp, F_sample, N_sample, T_chirp, K_chirp, F_LO, F_Cent_rx = self.get_chirp_params()
    
        #gets a frequency and Y_power fit based on -B_chirp/2 to B_chirp/2
        f_fit = f[self.find_index((F_Cent_rx-B_chirp/2)/1e6)[0]:self.find_index((F_Cent_rx+B_chirp/2)/1e6)[0]+1]
        Y_angle_fit = np.angle(Y[self.find_index((F_Cent_rx-B_chirp/2)/1e6)[0]:self.find_index((F_Cent_rx+B_chirp/2)/1e6)[0]+1])
        
        #deletes the middle point or 0 frequency term
        del_index = np.int(len(f_fit)/2)
        f_fit = np.delete(f_fit, del_index)
        Y_angle_fit = np.delete(Y_angle_fit, del_index)

        #fits the curve to a line and returns params -coefficients of the fit
        params, params_covariance = optimize.curve_fit(self.test_func, f_fit, Y_angle_fit, p0=[1, 0])

        #plots the curve
        if(plot_en):
            plt.plot(f_fit, Y_angle_fit)
            plt.plot(f_fit, self.test_func(f_fit, params[0], params[1]))
            plt.show()

        #returns the coefficients
        return params
    
    ###############################################################################
    # Returns the index corresponding to a value nearest freq_value in the array f.  f is found
    # from N_sample and F_sample.
    ###############################################################################
    def find_index(self, freq_value):
        
        #gets the global params from the class
        F_center, B_chirp, F_sample, N_sample, T_chirp, K_chirp, F_LO, F_Cent_rx = self.get_chirp_params()
        
        array = np.arange(-int(N_sample/2), int(N_sample/2))/N_sample*F_sample/1e6
        
        idx = (np.abs(array - freq_value)).argmin()
        return idx, array[idx]

    ###############################################################################
    # Removes the zero crossing and generates Y_angle_fit and f_fit.  
    ###############################################################################
    def remove_zero_cross(self, f, Y, B_chirp, F_Center):
        
        #gets the global params from the class
        F_center, B_chirp, F_sample, N_sample, T_chirp, K_chirp, F_LO, F_Cent_rx = self.get_chirp_params()
        
        f_fit = f[self.find_index((F_Cent_rx-B_chirp/2)/1e6)[0]:self.find_index((F_Cent_rx+B_chirp/2)/1e6)[0]+1]
        Y_angle_fit = np.angle(Y[self.find_index((F_Cent_rx-B_chirp/2)/1e6)[0]:self.find_index((F_Cent_rx+B_chirp/2)/1e6)[0]+1])
        del_index = np.int(len(f_fit)/2)

        f_fit = np.delete(f_fit, del_index)
        Y_angle_fit = np.delete(Y_angle_fit, del_index)

        return f_fit, Y_angle_fit
    
    ###############################################################################
    # returns an FFT.  data is the time series data from a chirp, F_sample is the sample frequency
    # of the ADRV9009 receiver, and window is the windowing function.
    ###############################################################################    
    def get_chirp_powerspectrum(self, data, F_sample, window):
        
        #gets the global params from the class
        F_center, B_chirp, F_sample, N_sample, T_chirp, K_chirp, F_LO, F_Cent_rx = self.get_chirp_params()

        #gets the fft of data*window such that the negative frequencies end up on the left
        X = np.fft.fftshift(scipy.fftpack.fft(np.fft.fftshift(data*window))) 
        
        #Normalize the Y data
        X_norm = (abs(X)*np.sqrt(K_chirp)/(F_sample*30000))**2
    
        #get freq array based on N_sample and F_sample
        xf = np.arange(-int(N_sample/2), int(N_sample/2))/N_sample*F_sample/1e6

        return xf, X_norm
    
    ###############################################################################
    # Gets a windowing function function based on data_in.  Data_in is a list of time series
    # chirp data.  The assumptions for data_in is that the chirp is centered and that the 
    # N_samples in data_in be 2x the N_samples corresponding to T_chirp.  This function
    # returns a list consisting of windowing function and a bunch of zeros to the left
    # and right.
    ###############################################################################    
    def getwindow(self, data_in):
        N_samples = int(len(data_in)*0.5)
        window = list(signal.tukey(N_samples, 0.5))
        
        N_zeros_left = int(len(data_in)*.25)
        N_zeros_right = int(len(data_in)) - N_samples - N_zeros_left
        
        zeros_left = [0]*N_zeros_left
        zeros_right = [0]*N_zeros_right
        window = zeros_left + window + zeros_right
        window = [32767*elem for elem in window]
        return window
    
    ###############################################################################
    # the line fit function used to fit Y_angle vs f to remove jitter.  f is a list of frequencies and 
    # the vars a and b are correspond to the line coefs.
    ###############################################################################   
    def test_func(self, f, a, b):
        
        F_sample = self.F_sample
        
        tau = 1/F_sample*a*1e6
        return -2*np.pi*f*tau +b

    ###############################################################################
    # main function of this classes processes pandas dataframe df containing time series data and 
    # applies averaging N_avg.  Returns incoherent average of FFT as a list of f and Y.
    ###############################################################################   
    def get_chirp_averages(self, df, N_avg, verbose):
        
        #gets the global params from the class
        F_center, B_chirp, F_sample, N_sample, T_chirp, K_chirp, F_LO, F_Cent_rx = self.get_chirp_params()

        #gets the shape of df.  N_sample is the length of the column and M_sample is the number of columns.
        N_sample = df.shape[0]
        M_sample = df.shape[1]
        
        self.N_sample = N_sample

        #checks if N_avg is valid
        if(N_avg > M_sample):
            print("N_avg invalid. N_avg cannot be larger than number of columns in the dataframe")
            return 

        #make an array for the frequency based on N_sample and F_Sample
        f = np.arange(-int(N_sample/2), int(N_sample/2))/N_sample*F_sample/1e6

        #get FFT of df
        df_FFT = pd.DataFrame()
        for i in df:
            df_FFT[i] = fftshift(fft(fftshift(df[i])))

        #get correlation between 1st fft and subsequent data points for time shifting
        df_corr = pd.DataFrame()
        for i in df_FFT:
            df_corr[i] = df_FFT[i].to_numpy()*np.conj(df_FFT[0].to_numpy())

        #fit the angle of the correlation to extract complex conjugate
        df_Xfix = pd.DataFrame()
        df_xfix = pd.DataFrame()
        for i in df_corr:
            #extract tau and offset term from the fit
            tau, offset = self.get_tau(f, df_corr[i], False)

            #time shift the original FFT by tau and offset
            df_Xfix[i] = df_FFT[i]*np.exp(1j*2*np.pi*f*tau-1j*offset) 

            #get the time domain data 
            df_xfix[i] = fftshift(ifft(fftshift(df_Xfix[i])))

        #check the correlation between the df_Xfix and the the original FFT data
        df_corrcheck = pd.DataFrame()
        params_list =[]

        plt.figure(figsize=(8,8))
        plt.grid(True)
        plt.xlim(-1*B_chirp/2*1e-6, B_chirp/2*1e-6)
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Correlation between Time shifted data and inital (degrees)')
        
        Y_angle_list = []
        
        for i in df:
            df_corrcheck[i] = df_Xfix[i]*np.conj(df_FFT[0])
            #params_list.append(fit_chirp(f, df_corrcheck[str(i)], B_chirp, F_center, False)[0])
            f_angle, Y_angle = self.remove_zero_cross(f, df_corrcheck[i], B_chirp, F_center) 
            plt.plot(f_angle, Y_angle/np.pi*180)
            Y_angle_list.append(Y_angle/np.pi*180)

        #plt.savefig("corrs.pdf", bbox_inches='tight')
        plt.show()

        #plot the difference between 1st and 2nd data points.  Should just be noise, if time shift worked
        if(verbose):
            plt.figure(figsize=(8,8))
            plt.ylabel("Counts")
            plt.xlabel("ADC Codes")
            plt.title("Fixed Data Histogram - difference between 1st and 2nd data points")
            plt.hist(np.real(df_xfix[0]) - np.real(df_xfix[1]))
            plt.yscale('log')
            plt.show()

        #same as above for the unfixed data
        if(verbose):
            plt.figure(figsize=(8,8))
            plt.ylabel("Counts")
            plt.xlabel("ADC Codes") 
            plt.title("Raw Data Histogram - difference between 1st and 2nd data points")
            plt.hist(np.real(df[0]) - np.real(df[1]))
            plt.yscale('log')
            plt.show()

        # get windowing function
        window = self.getwindow(df_xfix.to_numpy())

        #plot windowing function
        if(verbose):
            plt.figure(figsize = (8,8))
            plt.plot(np.real(df_xfix[0]), label = "Triggered Real Data")
            for i in range(1, M_sample):
                plt.plot(np.real(df_xfix[i]))
                         
            plt.plot(window, label = "windowing function")
            plt.xlabel("Number of Samples (arb)")
            plt.ylabel("Amplitude (ADC Code)")
            plt.legend()
            plt.show()

        #incoherent averaging of my time shifted (fixed) samples
        R_sum = 0
        for i in range(N_avg):
            x, R = self.get_chirp_powerspectrum(df_xfix[i], F_sample, window)
            R_sum += R

        R_avg = R_sum/N_avg
        
    #x_int = self.complex128_to_complexint(df_xfix)
        
    #put the fixed dataframe into table - aligned_data in the hdf5_datafile
    #self.save_array_data_to_hdf5(x_int, self.hdf5_datafile, 'aligned_data')
        
        return f, 10*np.log(R_avg)/np.log(10), df_xfix 
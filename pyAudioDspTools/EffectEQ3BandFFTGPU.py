from . import config
import cupy
#import matplotlib.pyplot as pyplot


"""########################################################################################
Creating a 3 Band FFT Equalizer class/device using the GPU.
The only difference to the non GPU version is the use of cupy instead of numpy.
If you want to learn, please check the non GPU version first.
Init Parameters: 
    lowshelf_frequency: Shelving frequency in Hertz [float] or [int] (for example 400.0)
    lowshelf_db: Gain in decibel [float] or [int] (for example 3.0 or -3.0)
    highshelf_frequency: Shelving frequency in Hertz [float] or [int] (for example 400 or 400.0)
    highshelf_db: Gain in decibel [float] or [int] (for example 3.0 or -3.0)

applyfilter
    Applies the filter to a 44100Hz/32 bit float signal of your choice.
    Should operate with values between -1.0 and 1.0
    
This class introduces latency equal to the value of config.chunk_size. 
Optimal operation with config.chunk_size=512
###########################################################################################"""


class CreateEQ3BandFFTGPU:
    """Creating a 3Band FFT EQ audio-effect class/device.

    Can be used to manipulate frequencies in your audio cupy-array.
    Is the faster one, the slower, non FFT based one being CreateEQ3Band.
    Is NOT overloaded with basic settings.
    This class introduces latency equal to config.config.chunk_size.

    Parameters
    ----------
    lowshelf_frequency : int or float
        Sets the frequency of the lowshelf-band in Hertz.
    lowshelf_db : int or float
        Increase or decrease the lows in decibel.
    midband_frequency : int or float
        Sets the frequency of the mid-band in Hertz. Has a fixed Q.
    midband_db : int or float
        Increase or decrease the selected mids in decibel.
    highshelf_frequency : int or float
        Sets the frequency of the highshelf-band in Hertz.
    highshelf_db : int or float
        Increase or decrease the highs in decibel.

    """
    def __init__(self,lowshelf_frequency,lowshelf_db,midband_frequency,midband_db,highshelf_frequency,highshelf_db):
        #Basic
        #config.chunk_size = config.chunk_size
        self.fS = config.sampling_rate  # Sampling rate.

        #Highshelf Properties
        self.fH_highshelf = highshelf_frequency
        self.highshelf_db = highshelf_db

        #Lowshelf Properties
        self.fH_lowshelf = lowshelf_frequency
        self.lowshelf_db = lowshelf_db

        #Lowshelf Properties
        self.fH_midband = midband_frequency
        self.midband_db = midband_db

        #Setting Kaiser-Windows properties
        self.filter_length = (config.chunk_size//2)-1 # Filter length, must be odd.
        self.array_slice_value_start = config.chunk_size + (self.filter_length // 2)
        self.array_slice_value_end =  config.chunk_size - (self.filter_length // 2)


        ################ Create Lowcut (Finally becomes Highshelf) Sinc Filter and FFT ##################
        # Compute sinc filter.
        self.sinc_filter_highshelf = cupy.sinc(
            2 * (self.fH_highshelf-self.fH_highshelf/4) / self.fS * (cupy.arange(self.filter_length) - (self.filter_length - 1) / 2))

        # Apply window.
        self.sinc_filter_highshelf *= cupy.kaiser(self.filter_length,6.0)

        # Normalize to get unity gain.
        self.sinc_filter_highshelf /= cupy.sum(self.sinc_filter_highshelf)

        #Spectral inversion to make lowcut out of highcut
        self.sinc_filter_highshelf = -self.sinc_filter_highshelf
        self.sinc_filter_highshelf[(self.filter_length - 1) // 2] += 1


        #Zero Padding the Sinc Filter to the length of the input array for easier processing
        #You don't need to use cupy.convolve when input-array and sinc-filter array are the same lenght, just multiply
        self.sinc_filter_highshelf = cupy.append(self.sinc_filter_highshelf, cupy.zeros(config.chunk_size - self.filter_length + 1))
        self.sinc_filter_highshelf = cupy.append(self.sinc_filter_highshelf, cupy.zeros(((len(self.sinc_filter_highshelf) * 2) - 3)))
        self.sinc_filter_highshelf = cupy.fft.fft(self.sinc_filter_highshelf)


        ################ Create Highcut (Finally becomes Lowshelf) Sinc Filter and FFT ##################
        # Compute sinc filter.
        self.sinc_filter_lowshelf = cupy.sinc(
            2 * (self.fH_lowshelf + self.fH_lowshelf/4) / self.fS * (cupy.arange(self.filter_length) - (self.filter_length - 1) / 2))

        # Apply window.
        self.sinc_filter_lowshelf *= cupy.kaiser(self.filter_length,6.0)

        # Normalize to get unity gain.
        self.sinc_filter_lowshelf /= cupy.sum(self.sinc_filter_lowshelf)

        #Zero Padding the Sinc Filter to the length of the input array
        self.sinc_filter_lowshelf = cupy.append(self.sinc_filter_lowshelf, cupy.zeros(config.chunk_size - self.filter_length + 1))
        self.sinc_filter_lowshelf = cupy.append(self.sinc_filter_lowshelf, cupy.zeros(((len(self.sinc_filter_lowshelf) * 2) - 3)))
        self.sinc_filter_lowshelf = cupy.fft.fft(self.sinc_filter_lowshelf)


        ################ Create Midband (Lowpass+Highpass) Sinc Filter and FFT ##################
        # Compute sinc filter.
        self.sinc_filter_mid_lowpass = cupy.sinc(
            2 * (self.fH_midband+self.fH_midband/4) / self.fS * (cupy.arange(self.filter_length) - (self.filter_length - 1) / 2))

        # Apply window.
        self.sinc_filter_mid_lowpass *= cupy.kaiser(self.filter_length,6.0)

        # Normalize to get unity gain.
        self.sinc_filter_mid_lowpass /= cupy.sum(self.sinc_filter_mid_lowpass)

        # Compute sinc filter.
        self.sinc_filter_mid_highpass = cupy.sinc(
            2 * (self.fH_midband-self.fH_midband/4) / self.fS * (cupy.arange(self.filter_length) - (self.filter_length - 1) / 2))

        # Apply window.
        self.sinc_filter_mid_highpass *= cupy.kaiser(self.filter_length,6.0)

        # Normalize to get unity gain.
        self.sinc_filter_mid_highpass /= cupy.sum(self.sinc_filter_mid_highpass)

        #Spectral inversion to make highpass
        self.sinc_filter_mid_highpass = -self.sinc_filter_mid_highpass
        self.sinc_filter_mid_highpass[(self.filter_length - 1) // 2] += 1

        #Zero Padding the Sinc Filter to the length of the input array
        self.sinc_filter_mid_lowpass = cupy.append(self.sinc_filter_mid_lowpass, cupy.zeros(config.chunk_size - self.filter_length + 1))
        self.sinc_filter_mid_lowpass = cupy.append(self.sinc_filter_mid_lowpass, cupy.zeros(((len(self.sinc_filter_mid_lowpass) * 2) - 3)))
        self.sinc_filter_mid_lowpass = cupy.fft.fft(self.sinc_filter_mid_lowpass)

        #Zero Padding the Sinc Filter to the length of the input array
        self.sinc_filter_mid_highpass = cupy.append(self.sinc_filter_mid_highpass, cupy.zeros(config.chunk_size - self.filter_length + 1))
        self.sinc_filter_mid_highpass = cupy.append(self.sinc_filter_mid_highpass, cupy.zeros(((len(self.sinc_filter_mid_highpass) * 2) - 3)))
        self.sinc_filter_mid_highpass = cupy.fft.fft(self.sinc_filter_mid_highpass)



        #Initializing arrays
        self.filtered_signal = cupy.zeros(config.chunk_size * 3)
        self.original_signal = cupy.zeros(config.chunk_size * 3)
        self.float32_array_input_1 = cupy.zeros(config.chunk_size)
        self.float32_array_input_2 = cupy.zeros(config.chunk_size)
        self.float32_array_input_3 = cupy.zeros(config.chunk_size)
        self.cut_size = cupy.int16((self.filter_length - 1) / 2)


    def apply(self, float32_array_input):
        """Applying the 3 Band FFT EQ to a cupy-array.

        Parameters
        ----------
        float32_array_input : float
            The array, which the effect should be applied on.

        Returns
        -------
        float
            The processed array, should be the exact same size as the input array

        """
        #Loading new chunk and replacing old ones
        self.float32_array_input_3 = self.float32_array_input_2
        self.float32_array_input_2 = self.float32_array_input_1
        self.float32_array_input_1 = float32_array_input

        self.original_signal = cupy.concatenate(
            (self.float32_array_input_3,self.float32_array_input_2,self.float32_array_input_1),axis=None)

        #FFT for transforming samples from time-domain to frequency-domain
        signal_fft = cupy.fft.fft(self.original_signal)

        #Highshelf processing
        filtered_signal_highshelf = signal_fft * self.sinc_filter_highshelf #applying sinc filter

        #Lowshelf processing FFT
        filtered_signal_lowshelf = signal_fft * self.sinc_filter_lowshelf #applying sinc filter

        #Midband processing
        filtered_signal_midband = signal_fft * (self.sinc_filter_mid_highpass * self.sinc_filter_mid_lowpass)#applying lowpass
        #filtered_signal_midband = filtered_signal_midband*(1/self.fS)
        #filtered_signal_midband = filtered_signal_midband * self.sinc_filter_mid_highpass #applying highpass to just get mid

        #Highshelf processing Time-Domain
        filtered_signal_highshelf = cupy.fft.ifft(filtered_signal_highshelf)
        filtered_signal_highshelf = filtered_signal_highshelf[self.array_slice_value_start:-self.array_slice_value_end]
        filtered_signal_highshelf = (filtered_signal_highshelf*(10**(self.highshelf_db/20))) - filtered_signal_highshelf

        #Lowshelf processing Time-Domain
        filtered_signal_lowshelf = cupy.fft.ifft(filtered_signal_lowshelf)
        filtered_signal_lowshelf = filtered_signal_lowshelf[self.array_slice_value_start:-self.array_slice_value_end]
        filtered_signal_lowshelf = (filtered_signal_lowshelf*(10**(self.lowshelf_db/20))) - filtered_signal_lowshelf

        #Midband processing Time-Domain
        filtered_signal_midband = cupy.fft.ifft(filtered_signal_midband)
        filtered_signal_midband = filtered_signal_midband[self.array_slice_value_start:-self.array_slice_value_end]
        filtered_signal_midband = (filtered_signal_midband*(10**(self.midband_db/20))) - filtered_signal_midband


        #Mixing signals
        float_array_output = filtered_signal_midband + self.float32_array_input_2 + filtered_signal_lowshelf + filtered_signal_highshelf

        return float_array_output.real.astype('float32')


        #xf = cupy.linspace(0.0, 1.0 / (2.0 * (1.0/44100.0)), 768)
        #fig, ax = pyplot.subplots()

        #ax.plot(xf, 2.0 / 1536 * cupy.abs(filtered_signal_highshelf[:1536 // 2]))
        #ax.plot(xf, 2.0 / 1536 * cupy.abs(signal_fft[:1536 // 2]))
        #pyplot.show()



        #xf = cupy.linspace(0.0, 1.0 / (2.0 * (1.0/44100.0)), 768)
        #fig, ax = pyplot.subplots()

        #ax.plot(xf, 2.0 / 1536 * cupy.abs(add_signal_lowshelf[:1536 // 2]))
        #ax.plot(xf, 2.0 / 1536 * cupy.abs(filtered_signal[:1536 // 2]))
        #pyplot.show()
        # pyplot.plot(filtered_signal_highshelf)
        # pyplot.plot(self.float32_array_input_2)
        # pyplot.show()

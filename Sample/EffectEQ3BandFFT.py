import numpy
import config
#import matplotlib.pyplot as pyplot


"""########################################################################################
Creating a 3 Band FFT Equalizer class/device.
Init Parameters: 
    lowshelf_frequency: Shelving frequency in Hertz [float] or [int] (for example 400.0)
    lowshelf_db: Gain in decibel [float] or [int] (for example 3.0 or -3.0)
    highshelf_frequency: Shelving frequency in Hertz [float] or [int] (for example 400 or 400.0)
    highshelf_db: Gain in decibel [float] or [int] (for example 3.0 or -3.0)

applyfilter
    Applies the filter to a 44100Hz/32 bit float signal of your choice.
    Should operate with values between -1.0 and 1.0
    
This class introduces latency equal to the value of chunk_size. 
Optimal operation with chunk_size=512
###########################################################################################"""


class CreateEQ3BandFFT:
    def __init__(self,lowshelf_frequency,lowshelf_db,midband_frequency,midband_db,highshelf_frequency,highshelf_db):
        #Basic
        chunk_size = config.chunk_size
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
        self.filter_length = (chunk_size//2)-1 # Filter length, must be odd.
        self.array_slice_value_start = chunk_size + (self.filter_length // 2)
        self.array_slice_value_end =  chunk_size - (self.filter_length // 2)


        ################ Create Lowcut (Finally becomes Highshelf) Sinc Filter and FFT ##################
        # Compute sinc filter.
        self.sinc_filter_highshelf = numpy.sinc(
            2 * (self.fH_highshelf-self.fH_highshelf/4) / self.fS * (numpy.arange(self.filter_length) - (self.filter_length - 1) / 2))

        # Apply window.
        self.sinc_filter_highshelf *= numpy.kaiser(self.filter_length,6.0)

        # Normalize to get unity gain.
        self.sinc_filter_highshelf /= numpy.sum(self.sinc_filter_highshelf)

        #Spectral inversion to make lowcut out of highcut
        self.sinc_filter_highshelf = -self.sinc_filter_highshelf
        self.sinc_filter_highshelf[(self.filter_length - 1) // 2] += 1


        #Zero Padding the Sinc Filter to the length of the input array for easier processing
        #You don't need to use numpy.convolve when input-array and sinc-filter array are the same lenght, just multiply
        self.sinc_filter_highshelf = numpy.append(self.sinc_filter_highshelf, numpy.zeros(chunk_size - self.filter_length + 1))
        self.sinc_filter_highshelf = numpy.append(self.sinc_filter_highshelf, numpy.zeros(((len(self.sinc_filter_highshelf) * 2) - 3)))
        self.sinc_filter_highshelf = numpy.fft.fft(self.sinc_filter_highshelf)


        ################ Create Highcut (Finally becomes Lowshelf) Sinc Filter and FFT ##################
        # Compute sinc filter.
        self.sinc_filter_lowshelf = numpy.sinc(
            2 * (self.fH_lowshelf + self.fH_lowshelf/4) / self.fS * (numpy.arange(self.filter_length) - (self.filter_length - 1) / 2))

        # Apply window.
        self.sinc_filter_lowshelf *= numpy.kaiser(self.filter_length,6.0)

        # Normalize to get unity gain.
        self.sinc_filter_lowshelf /= numpy.sum(self.sinc_filter_lowshelf)

        #Zero Padding the Sinc Filter to the length of the input array
        self.sinc_filter_lowshelf = numpy.append(self.sinc_filter_lowshelf, numpy.zeros(chunk_size - self.filter_length + 1))
        self.sinc_filter_lowshelf = numpy.append(self.sinc_filter_lowshelf, numpy.zeros(((len(self.sinc_filter_lowshelf) * 2) - 3)))
        self.sinc_filter_lowshelf = numpy.fft.fft(self.sinc_filter_lowshelf)


        ################ Create Midband (Lowpass+Highpass) Sinc Filter and FFT ##################
        # Compute sinc filter.
        self.sinc_filter_mid_lowpass = numpy.sinc(
            2 * (self.fH_midband+self.fH_midband/4) / self.fS * (numpy.arange(self.filter_length) - (self.filter_length - 1) / 2))

        # Apply window.
        self.sinc_filter_mid_lowpass *= numpy.kaiser(self.filter_length,6.0)

        # Normalize to get unity gain.
        self.sinc_filter_mid_lowpass /= numpy.sum(self.sinc_filter_mid_lowpass)

        # Compute sinc filter.
        self.sinc_filter_mid_highpass = numpy.sinc(
            2 * (self.fH_midband-self.fH_midband/4) / self.fS * (numpy.arange(self.filter_length) - (self.filter_length - 1) / 2))

        # Apply window.
        self.sinc_filter_mid_highpass *= numpy.kaiser(self.filter_length,6.0)

        # Normalize to get unity gain.
        self.sinc_filter_mid_highpass /= numpy.sum(self.sinc_filter_mid_highpass)

        #Spectral inversion to make highpass
        self.sinc_filter_mid_highpass = -self.sinc_filter_mid_highpass
        self.sinc_filter_mid_highpass[(self.filter_length - 1) // 2] += 1

        #Zero Padding the Sinc Filter to the length of the input array
        self.sinc_filter_mid_lowpass = numpy.append(self.sinc_filter_mid_lowpass, numpy.zeros(chunk_size - self.filter_length + 1))
        self.sinc_filter_mid_lowpass = numpy.append(self.sinc_filter_mid_lowpass, numpy.zeros(((len(self.sinc_filter_mid_lowpass) * 2) - 3)))
        self.sinc_filter_mid_lowpass = numpy.fft.fft(self.sinc_filter_mid_lowpass)

        #Zero Padding the Sinc Filter to the length of the input array
        self.sinc_filter_mid_highpass = numpy.append(self.sinc_filter_mid_highpass, numpy.zeros(chunk_size - self.filter_length + 1))
        self.sinc_filter_mid_highpass = numpy.append(self.sinc_filter_mid_highpass, numpy.zeros(((len(self.sinc_filter_mid_highpass) * 2) - 3)))
        self.sinc_filter_mid_highpass = numpy.fft.fft(self.sinc_filter_mid_highpass)



        #Initializing arrays
        self.filtered_signal = numpy.zeros(chunk_size * 3)
        self.original_signal = numpy.zeros(chunk_size * 3)
        self.float32_array_input_1 = numpy.zeros(chunk_size)
        self.float32_array_input_2 = numpy.zeros(chunk_size)
        self.float32_array_input_3 = numpy.zeros(chunk_size)
        self.cut_size = numpy.int16((self.filter_length - 1) / 2)


    def apply(self, float32_array_input):
        #Loading new chunk and replacing old ones
        self.float32_array_input_3 = self.float32_array_input_2
        self.float32_array_input_2 = self.float32_array_input_1
        self.float32_array_input_1 = float32_array_input

        self.original_signal = numpy.concatenate(
            (self.float32_array_input_3,self.float32_array_input_2,self.float32_array_input_1),axis=None)

        #FFT for transforming samples from time-domain to frequency-domain
        signal_fft = numpy.fft.fft(self.original_signal)

        #Highshelf processing
        filtered_signal_highshelf = signal_fft * self.sinc_filter_highshelf #applying sinc filter

        #Lowshelf processing FFT
        filtered_signal_lowshelf = signal_fft * self.sinc_filter_lowshelf #applying sinc filter

        #Midband processing
        filtered_signal_midband = signal_fft * (self.sinc_filter_mid_highpass * self.sinc_filter_mid_lowpass)#applying lowpass
        #filtered_signal_midband = filtered_signal_midband*(1/self.fS)
        #filtered_signal_midband = filtered_signal_midband * self.sinc_filter_mid_highpass #applying highpass to just get mid

        #Highshelf processing Time-Domain
        filtered_signal_highshelf = numpy.fft.ifft(filtered_signal_highshelf)
        filtered_signal_highshelf = filtered_signal_highshelf[self.array_slice_value_start:-self.array_slice_value_end]
        filtered_signal_highshelf = (filtered_signal_highshelf*(10**(self.highshelf_db/20))) - filtered_signal_highshelf

        #Lowshelf processing Time-Domain
        filtered_signal_lowshelf = numpy.fft.ifft(filtered_signal_lowshelf)
        filtered_signal_lowshelf = filtered_signal_lowshelf[self.array_slice_value_start:-self.array_slice_value_end]
        filtered_signal_lowshelf = (filtered_signal_lowshelf*(10**(self.lowshelf_db/20))) - filtered_signal_lowshelf

        #Midband processing Time-Domain
        filtered_signal_midband = numpy.fft.ifft(filtered_signal_midband)
        filtered_signal_midband = filtered_signal_midband[self.array_slice_value_start:-self.array_slice_value_end]
        filtered_signal_midband = (filtered_signal_midband*(10**(self.midband_db/20))) - filtered_signal_midband


        #Mixing signals
        float_array_output = filtered_signal_midband + self.float32_array_input_2 + filtered_signal_lowshelf + filtered_signal_highshelf

        return float_array_output.real.astype('float32')


        #xf = numpy.linspace(0.0, 1.0 / (2.0 * (1.0/44100.0)), 768)
        #fig, ax = pyplot.subplots()

        #ax.plot(xf, 2.0 / 1536 * numpy.abs(filtered_signal_highshelf[:1536 // 2]))
        #ax.plot(xf, 2.0 / 1536 * numpy.abs(signal_fft[:1536 // 2]))
        #pyplot.show()



        #xf = numpy.linspace(0.0, 1.0 / (2.0 * (1.0/44100.0)), 768)
        #fig, ax = pyplot.subplots()

        #ax.plot(xf, 2.0 / 1536 * numpy.abs(add_signal_lowshelf[:1536 // 2]))
        #ax.plot(xf, 2.0 / 1536 * numpy.abs(filtered_signal[:1536 // 2]))
        #pyplot.show()
        # pyplot.plot(filtered_signal_highshelf)
        # pyplot.plot(self.float32_array_input_2)
        # pyplot.show()

import config
import numpy
import matplotlib.pyplot as pyplot


"""########################################################################################
Creating an FFT Lowcut or Highcut Filter.
Init Parameters: 
    cutoff frequency: Cutoff frequency in Hertz [float] or [int] (for example 400.0)

applyfilter
    Applies the filter to a 44100Hz/32 bit float signal of your choice.
    Should operate with values between -1.0 and 1.0

This class introduces latency equal to the value of chunk_size found in config.py. 
Optimal operation with chunk_size=512
###########################################################################################"""


class CreateHighCutFilter:
    def __init__(self,cutoff_frequency):
        chunk_size = config.chunk_size
        self.fS = config.sampling_rate  # Sampling rate.
        self.fH = cutoff_frequency # Cutoff frequency.
        self.filter_length = (chunk_size//2)-1 #Filter length, must be odd.

        self.array_slice_value_start = chunk_size + (self.filter_length // 2)
        self.array_slice_value_end = chunk_size - (self.filter_length // 2)


        # Compute sinc filter.
        self.sinc_filter = numpy.sinc(2 * self.fH / self.fS * (numpy.arange(self.filter_length) - (self.filter_length - 1) / 2))
        #pyplot.plot(self.sinc_filter)

        # Apply window.
        self.sinc_filter *= numpy.blackman(self.filter_length)
        #pyplot.plot(self.sinc_filter)

        # Normalize to get unity gain.
        self.sinc_filter /= numpy.sum(self.sinc_filter)
        #print(len(self.sinc_filter))

        self.filtered_signal = numpy.zeros(chunk_size*3)
        self.float32_array_input_1 = numpy.zeros(chunk_size)
        self.float32_array_input_2 = numpy.zeros(chunk_size)
        self.float32_array_input_3 = numpy.zeros(chunk_size)


        self.cut_size = numpy.int16((self.filter_length-1)/2)
        self.sinc_filter = numpy.append(self.sinc_filter,numpy.zeros(chunk_size - self.filter_length + 1))
        self.sinc_filter = numpy.append(self.sinc_filter,numpy.zeros(((len(self.sinc_filter)*2)-3)))
        self.sinc_filter = numpy.fft.fft(self.sinc_filter)

    def apply(self,float32_array_input):
        self.float32_array_input_3 = self.float32_array_input_2
        self.float32_array_input_2 = self.float32_array_input_1
        self.float32_array_input_1 = float32_array_input

        self.filtered_signal = numpy.concatenate(
            (self.float32_array_input_3,self.float32_array_input_2,self.float32_array_input_1),axis=None)

        self.filtered_signal = numpy.fft.fft(self.filtered_signal)
        self.filtered_signal = self.filtered_signal * self.sinc_filter
        self.filtered_signal = numpy.fft.ifft(self.filtered_signal)
        self.filtered_signal = self.filtered_signal[self.array_slice_value_start:-self.array_slice_value_end]

        return self.filtered_signal.real.astype('float32')




class CreateLowCutFilter:
    def __init__(self,cutoff_frequency):
        chunk_size = config.chunk_size
        self.fS = config.sampling_rate  # Sampling rate.
        self.fH = cutoff_frequency  # Cutoff frequency.
        self.filter_length = (chunk_size//2)-1  # Filter length, must be odd.

        self.array_slice_value_start = chunk_size + (self.filter_length // 2)
        self.array_slice_value_end = chunk_size - (self.filter_length // 2)

        # Compute sinc filter.
        self.sinc_filter = numpy.sinc(
            2 * self.fH / self.fS * (numpy.arange(self.filter_length) - (self.filter_length - 1) / 2))

        # Apply window.
        self.sinc_filter *= numpy.blackman(self.filter_length)

        # Normalize to get unity gain.
        self.sinc_filter /= numpy.sum(self.sinc_filter)
        #print(len(self.sinc_filter))

        #Spectral inversion to create Lowcut from Highcut
        self.sinc_filter = -self.sinc_filter
        self.sinc_filter[(self.filter_length - 1) // 2] += 1

        self.filtered_signal = numpy.zeros(chunk_size * 3)
        self.float32_array_input_1 = numpy.zeros(chunk_size)
        self.float32_array_input_2 = numpy.zeros(chunk_size)
        self.float32_array_input_3 = numpy.zeros(chunk_size)

        self.cut_size = numpy.int16((self.filter_length - 1) / 2)
        self.sinc_filter = numpy.append(self.sinc_filter, numpy.zeros(chunk_size - self.filter_length + 1))
        self.sinc_filter = numpy.append(self.sinc_filter, numpy.zeros(((len(self.sinc_filter) * 2) - 3)))
        self.sinc_filter = numpy.fft.fft(self.sinc_filter)


    def apply(self, float32_array_input):
        self.float32_array_input_3 = self.float32_array_input_2
        self.float32_array_input_2 = self.float32_array_input_1
        self.float32_array_input_1 = float32_array_input

        self.filtered_signal = numpy.concatenate(
            (self.float32_array_input_3, self.float32_array_input_2, self.float32_array_input_1), axis=None)

        self.filtered_signal = numpy.fft.fft(self.filtered_signal)
        self.filtered_signal = (self.filtered_signal * self.sinc_filter)
        self.filtered_signal = numpy.fft.ifft(self.filtered_signal)
        self.filtered_signal = self.filtered_signal[self.array_slice_value_start:-self.array_slice_value_end]

        return self.filtered_signal.real.astype('float32')

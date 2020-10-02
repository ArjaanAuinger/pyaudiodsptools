from __future__ import print_function
from __future__ import division

import numpy
import matplotlib.pyplot as pyplot

# Example code, computes the coefficients of a high-pass windowed-sinc filter.

class CreateLowCutFilter:
    def __init__(self):
        self.fS = 44100  # Sampling rate.
        self.fH = 3000 # Cutoff frequency.
        self.filter_length = 513#391  # Filter length, must be odd.


        # Compute sinc filter.
        self.sinc_filter = numpy.sinc(2 * self.fH / self.fS * (numpy.arange(self.filter_length) - (self.filter_length - 1) / 2))
        pyplot.plot(self.sinc_filter)

        # Apply window.
        self.sinc_filter *= numpy.hamming(self.filter_length)
        pyplot.plot(self.sinc_filter)

        # Normalize to get unity gain.
        self.sinc_filter /= numpy.sum(self.sinc_filter)
        #self.sinc_filter=self.sinc_filter[1:-1]
        print(len(self.sinc_filter))

        self.overlap_array = numpy.zeros(self.filter_length)
        self.keep_end = numpy.zeros(512)


        self.first_chunk = True
        self.cut_size = numpy.int16((self.filter_length-1)/2)
        self.last_sample = 0.0
        self.filtered_signal_2 = numpy.zeros(1060,dtype='float32')
        self.filtered_signal_1 = numpy.zeros(1060,dtype='float32')
        self.filter_save = numpy.zeros(16,dtype='float32')
        #self.counter = 1
        self.sinc_filter = numpy.append(self.sinc_filter,numpy.zeros(len(self.sinc_filter)-2))
        self.sinc_filter = numpy.fft.fft(self.sinc_filter)

        pyplot.plot(self.sinc_filter)
        #pyplot.plot(numpy.blackman(self.filter_length))
        pyplot.show()

        # Create a high-pass filter from the low-pass filter through spectral inversion.
        #self.sinc_filter = -self.sinc_filter
        #self.sinc_filter[(self.filter_length - 1) // 2] += 1

    def applyfilter(self,float32_array_input):
        float32_array_input = numpy.append(float32_array_input,numpy.zeros(len(float32_array_input)))
        #float32_array_input[0] = self.last_sample
        #self.last_sample = float32_array_input[-1:]
        #float32_array_input = numpy.pad(float32_array_input,2,mode='edge')
        # Applying the filter to a signal s can be as simple as writing
        #filtered_signal_3 = numpy.convolve(float32_array_input, self.sinc_filter,mode='same')
        filtered_signal_3 = numpy.fft.fft(float32_array_input)
        filtered_signal_3 = filtered_signal_3 * self.sinc_filter
        filtered_signal_3 = numpy.fft.ifft(filtered_signal_3)
        filtered_signal_3 = filtered_signal_3[256:-256]
        #filtered_signal_3[:512] = filtered_signal_3[:512] + self.keep_end
        #self.keep_end = filtered_signal_3[-512:]
        #filtered_signal_3 = filtered_signal_3[:512]
        #float32_array_input[:16] = self.filter_save
        #filtered_signal_3 = filtered_signal_3[self.cut_size:-(self.cut_size)]
        #self.filter_save = filtered_signal_3[-16:]
        return filtered_signal_3
        """
        filtered_signal_2 = self.filtered_signal_2
        filtered_signal_1 = self.filtered_signal_1

        float32_array_output = (filtered_signal_1[-self.cut_size:]+filtered_signal_2[:self.cut_size])
        float32_array_output = numpy.append(float32_array_output,filtered_signal_2[self.cut_size:-self.cut_size])
        float32_array_output = numpy.append(float32_array_output,filtered_signal_2[-self.cut_size:] + filtered_signal_3[:self.cut_size])

        float32_array_output = float32_array_output[self.cut_size:-self.cut_size]
        self.filtered_signal_2 = filtered_signal_3
        self.filtered_signal_1 = filtered_signal_2
        return float32_array_output
        """








        #filtered_signal[0] = self.last_sample
        #if self.first_chunk == True:
            #filtered_signal = filtered_signal[(self.cut_size):]

        #filtered_signal = filtered_signal[0:-numpy.int16(self.filter_length-1)]

        #filtered_signal = filtered_signal = filtered_signal[(self.cut_size+self.filter_length):-(self.cut_size+self.filter_length)]

        #filtered_signal[0:self.filter_length] = filtered_signal[0:self.filter_length] + self.overlap_array
        #self.overlap_array = filtered_signal[-20:]
        #self.counter += 1
        #print(len(filtered_signal))
        #return filtered_signal

#als erstes overlappen
#dann ende ins self.overlap array laden
#ende löschen

"""
from __future__ import print_function
from __future__ import division

import numpy as np

# Example code, computes the coefficients of a low-pass windowed-sinc filter.

# Configuration.
fS = 44100  # Sampling rate.
fL = 4410  # Cutoff frequency.
N = 39  # Filter length, must be odd.

# Compute sinc filter.
h = np.sinc(2 * fL / fS * (np.arange(N) - (N - 1) / 2))

# Apply window.
h *= np.hamming(N)

# Normalize to get unity gain.
h /= np.sum(h)

print(h)

# Applying the filter to a signal s can be as simple as writing
# s = np.convolve(s, h)
"""
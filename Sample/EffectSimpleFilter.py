from __future__ import print_function
from __future__ import division

import numpy
import matplotlib.pyplot as pyplot

# Example code, computes the coefficients of a high-pass windowed-sinc filter.

class CreateHighCutFilter:
    def __init__(self,cutoff_frequency,chunk_size):
        self.fS = 44100  # Sampling rate.
        self.fH = cutoff_frequency # Cutoff frequency.
        self.filter_length = chunk_size + 1 #Filter length, must be odd.


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
        self.sinc_filter = numpy.append(self.sinc_filter,numpy.zeros(((len(self.sinc_filter)*2)-3)))
        self.sinc_filter = numpy.fft.fft(self.sinc_filter)

    def applyfilter(self,float32_array_input):
        self.float32_array_input_3 = self.float32_array_input_2
        self.float32_array_input_2 = self.float32_array_input_1
        self.float32_array_input_1 = float32_array_input

        self.filtered_signal = numpy.concatenate(
            (self.float32_array_input_3,self.float32_array_input_2,self.float32_array_input_1),axis=None)

        self.filtered_signal = numpy.fft.fft(self.filtered_signal)
        self.filtered_signal = self.filtered_signal * self.sinc_filter
        self.filtered_signal = numpy.fft.ifft(self.filtered_signal)
        self.filtered_signal = self.filtered_signal[512:-512]

        return self.filtered_signal




class CreateLowCutFilter:
    def __init__(self,cutoff_frequency,chunk_size):
        self.fS = 44100  # Sampling rate.
        self.fH = cutoff_frequency  # Cutoff frequency.
        self.filter_length = chunk_size + 1  # Filter length, must be odd.

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
        self.sinc_filter = numpy.append(self.sinc_filter, numpy.zeros(((len(self.sinc_filter) * 2) - 3)))
        self.sinc_filter = numpy.fft.fft(self.sinc_filter)


    def applyfilter(self, float32_array_input):
        self.float32_array_input_3 = self.float32_array_input_2
        self.float32_array_input_2 = self.float32_array_input_1
        self.float32_array_input_1 = float32_array_input

        self.filtered_signal = numpy.concatenate(
            (self.float32_array_input_3, self.float32_array_input_2, self.float32_array_input_1), axis=None)

        self.filtered_signal = numpy.fft.fft(self.filtered_signal)
        self.filtered_signal = (self.filtered_signal * self.sinc_filter)
        self.filtered_signal = numpy.fft.ifft(self.filtered_signal)
        self.filtered_signal = self.filtered_signal[512:-512]

        return self.filtered_signal

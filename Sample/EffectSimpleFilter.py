from __future__ import print_function
from __future__ import division

import numpy
import matplotlib.pyplot as pyplot

# Example code, computes the coefficients of a high-pass windowed-sinc filter.

class CreateLowCutFilter:
    def __init__(self):
        self.fS = 44100  # Sampling rate.
        self.fH = 2000 # Cutoff frequency.
        self.filter_length = 37#391  # Filter length, must be odd.

        # Compute sinc filter.
        self.sinc_filter = numpy.sinc(2 * self.fH / self.fS * (numpy.arange(self.filter_length) - (self.filter_length - 1) / 2))

        # Apply window.
        self.sinc_filter *= numpy.blackman(self.filter_length)

        # Normalize to get unity gain.
        self.sinc_filter /= numpy.sum(self.sinc_filter)
        print(len(self.sinc_filter))

        self.overlap_array = numpy.zeros(self.filter_length)

        pyplot.plot(self.sinc_filter)
        pyplot.plot(numpy.blackman(self.filter_length))
        pyplot.show()

        # Create a high-pass filter from the low-pass filter through spectral inversion.
        #self.sinc_filter = -self.sinc_filter
        #self.sinc_filter[(self.filter_length - 1) // 2] += 1

    def applyfilter(self,float32_array_input):
        # Applying the filter to a signal s can be as simple as writing
        filtered_signal = numpy.convolve(float32_array_input, self.sinc_filter)
        filtered_signal[0:numpy.int16(self.filter_length)] = filtered_signal[0:numpy.int16(self.filter_length)] + self.overlap_array
        self.overlap_array = filtered_signal[-numpy.int16(self.filter_length):]
        filtered_signal = filtered_signal[0:-numpy.int16(self.filter_length)]
        #filtered_signal = filtered_signal[numpy.int16(self.filter_length/2):-numpy.int16(self.filter_length/2)]
        print(len(filtered_signal))
        return filtered_signal

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
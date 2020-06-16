#!/usr/bin/env python3
import os
import sys
import time
import timeit
import numpy
import math
import matplotlib.pyplot as pyplot
import scipy
import copy

#512 samples@44.1 kHz = 11.7ms = 0.00117s
print(numpy.finfo('float64').max)# 1.79
print(numpy.finfo('float64').min)# -1.79

def createsinewave(sin_sample_rate, sin_frequency, sin_buffer_size):
    sin_time_array = numpy.arange(sin_buffer_size)
    sin_amplitude_array = numpy.sin(2 * numpy.pi * sin_frequency*sin_time_array/sin_sample_rate)
    return (sin_time_array, sin_amplitude_array)


def createsquarewave(square_sample_rate, square_frequency, square_buffer_size):
    square_time_array = numpy.arange(square_buffer_size)
    square_amplitude_array = numpy.sin(2 * numpy.pi * square_frequency * square_time_array / square_sample_rate)
    #print (square_amplitude_array)
    for sample in square_time_array:
            if square_amplitude_array[sample] > 0:
                square_amplitude_array[sample] = 1.0
            if square_amplitude_array[sample] < 0:
                square_amplitude_array[sample] = -1.0
    #print (square_amplitude_array)
    return (square_time_array, square_amplitude_array)
"""
def createwhitenoise(whitenoise_buffer_size):
    samples = numpy.random.normal(0, 1, size=whitenoise_buffer_size)
    print (samples)
"""

def createwhitenoise(buffer_size, sample_rate):
    whitenoise_time_array = numpy.arange(buffer_size)
    freqs = numpy.abs(numpy.fft.fftfreq(buffer_size, 1/sample_rate))
    f = numpy.zeros(buffer_size)
    idx = numpy.where(numpy.logical_and(freqs>=20, freqs<=20000))[0]
    f[idx] = 1

    def fftnoise(f):
        f = numpy.array(f, dtype='complex')
        Np = (len(f) - 1) // 2
        phases = numpy.random.rand(Np) * 2 * numpy.pi
        phases = numpy.cos(phases) + 1j * numpy.sin(phases)
        f[1:Np + 1] *= phases
        f[-1:-1 - Np:-1] = numpy.conj(f[1:Np + 1])
        return (numpy.fft.ifft(f).real*5)

    whitenoise_amplitude_array = fftnoise(f)

    return whitenoise_time_array,whitenoise_amplitude_array


def converfloatto16bit(float_array_input):
    float_array_input = numpy.where(float_array_input < 1, float_array_input, 1)
    print(float_array_input)
    float_array_input = numpy.where(float_array_input > -1, float_array_input, -1)

    float_array_output = numpy.int16(float_array_input * (2 ** 15 - 1))
    return float_array_output

def dBVinfo(float_array_input):
    count = 0
    added_sample_value = 0.0
    average_sample_value = 0.0
    audio_array = copy.deepcopy(float_array_input)
    audio_array = numpy.where(audio_array > 0, audio_array, audio_array * (-1))
    average_sample_value = audio_array.sum()/audio_array.size
    amplitude = average_sample_value/1
    dBV = 20 * math.log10(amplitude)
    return dBV

def dB16bitinfo(float_array_input):
    count = 0
    added_sample_value = 0.0
    average_sample_value = 0.0
    audio_array = copy.deepcopy(float_array_input)
    audio_array = numpy.where(audio_array > 0, audio_array, audio_array * (-1))
    average_sample_value = audio_array.sum()/audio_array.size
    amplitude = average_sample_value/32767
    dB16 = 20 * math.log10(amplitude)
    return dB16


x,y = createwhitenoise(512,44100)

x2,y2 = createsinewave(44100,1000,512)

x3,y3 = createsquarewave(44100,1000,512)



start = timeit.default_timer()
y2=converfloatto16bit(y2)
stop = timeit.default_timer()
print('Time: ', (stop - start)*1000, 'ms')



pyplot.plot(x, y)
pyplot.plot(x2,y2)
pyplot.plot(x3,y3)
#pyplot.plot(x2,y2)

pyplot.show()



sys.exit()



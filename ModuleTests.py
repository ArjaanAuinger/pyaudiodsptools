#!/usr/bin/env python3
# matplotlib for debugging
"""
Module Testing for pyaudiodsptools. Just $pip install pyAudioDspTools.
Then create a scratch and run this script.
"""



#import matplotlib.pyplot as pyplot

import pyAudioDspTools

import os
import sys
import time
import timeit
import numpy
import math
import copy
import wave
import struct

# 512 samples@44.1 kHz = 11.7ms = 0.00117s
# print(numpy.finfo('float64').max)# 1.79
# print(numpy.finfo('float64').min)# -1.79
# print(numpy.iinfo('int32').max)# 32767
# print(numpy.iinfo('int32').min)# -32768
# print(numpy.iinfo('int16').max)# 32767
# print(numpy.iinfo('int16').min)# -32768
# print(numpy.iinfo('int8').max)# 127
# print(numpy.iinfo('int8').min)# -128

pyAudioDspTools.config.initialize(44100, 512)

from pyAudioDspTools import config

from pyAudioDspTools.Generators import CreateSinewave, CreateSquarewave, CreateWhitenoise
from pyAudioDspTools.Utility import MakeChunks, CombineChunks, MixSignals, ConvertdBVTo16Bit
from pyAudioDspTools.Utility import Convert16BitTodBV, Dither16BitTo8Bit, Dither32BitIntTo16BitInt, MonoWavToNumpyFloat, InfodBV
from pyAudioDspTools.Utility import InfodBV16Bit, VolumeChange, MonoWavToNumpy16BitInt, NumpyFloatToWav
from pyAudioDspTools.EffectCompressor import CreateCompressor
from pyAudioDspTools.EffectGate import CreateGate
from pyAudioDspTools.EffectDelay import CreateDelay
from pyAudioDspTools._EffectReverb import CreateReverb
from pyAudioDspTools.EffectFFTFilter import CreateHighCutFilter, CreateLowCutFilter
from pyAudioDspTools.EffectEQ3BandFFT import CreateEQ3BandFFT
from pyAudioDspTools.EffectEQ3Band import CreateEQ3Band
from pyAudioDspTools.EffectSoftClipper import CreateSoftClipper
from pyAudioDspTools.EffectHardDistortion import CreateHardDistortion
from pyAudioDspTools.EffectTremolo import CreateTremolo
from pyAudioDspTools.EffectSaturator import CreateSaturator

test_length_in_samples = 44100*60

print('####Creating Generators####')
sine_full = CreateSinewave(1000, test_length_in_samples)
print('Sine Wave ok...')
square_full = CreateSquarewave(1000, test_length_in_samples)
print('Square Wave ok...')
noise_full = CreateWhitenoise(test_length_in_samples)
print('White Noise ok..-')
print('')


print('####Making Copies####')
sine_copy = copy.deepcopy(sine_full)
sine_chunked = MakeChunks(sine_copy)
print('done.')
print('')

print('####Creating Effect Classes####')
harddistortiontest = CreateHardDistortion()
tremolotest = CreateTremolo()
delaytest = CreateDelay()
compressortest = CreateCompressor()
softclippertest = CreateSoftClipper()
saturatortest = CreateSaturator()
gatetest = CreateGate()
reverbtest = CreateReverb()
lowcuttest = CreateLowCutFilter(200)
highcuttest = CreateHighCutFilter(8000)
eq3bandtest = CreateEQ3Band(100, 2, 700, -4, 8000, 5)
eq3bandffttest = CreateEQ3BandFFT(100, 2, 700, -4, 8000, 5)
print('done...')
print('')

print('####config.py settings####')
print('Sample Rate is: ', config.sampling_rate, 'Hz')
print('Chunk Size / Buffer Size is: ', config.chunk_size, 'samples')
print('GPU is available: ', config._gpu_available)
print('GPU is used: ', config.use_gpu)
print('')


print('####Testing Saturator####')
start = time.perf_counter()
counter = 0
for counter in range(len(sine_chunked)):
    sine_chunked[counter] = saturatortest.apply(sine_chunked[counter])
    counter += 1
stop = time.perf_counter()
print('Success!')
print('Total Time: ', (stop - start) * 1000, 'ms')
print('Time per Chunk', ((stop - start) * 1000) / len(sine_chunked), 'ms')
print('')

print('####Testing Compressor####')
start = time.perf_counter()
counter = 0
for counter in range(len(sine_chunked)):
    sine_chunked[counter] = compressortest.apply(sine_chunked[counter])
    counter += 1
stop = time.perf_counter()
print('Success!')
print('Total Time: ', (stop - start) * 1000, 'ms')
print('Time per Chunk', ((stop - start) * 1000) / len(sine_chunked), 'ms')
print('')

print('####Testing Delay####')
start = time.perf_counter()
counter = 0
for counter in range(len(sine_chunked)):
    sine_chunked[counter] = delaytest.apply(sine_chunked[counter])
    counter += 1
stop = time.perf_counter()
print('Success!')
print('Total Time: ', (stop - start) * 1000, 'ms')
print('Time per Chunk', ((stop - start) * 1000) / len(sine_chunked), 'ms')
print('')

print('####Testing Tremolo####')
start = time.perf_counter()
counter = 0
for counter in range(len(sine_chunked)):
    sine_chunked[counter] = tremolotest.apply(sine_chunked[counter])
    counter += 1
stop = time.perf_counter()
print('Success!')
print('Total Time: ', (stop - start) * 1000, 'ms')
print('Time per Chunk', ((stop - start) * 1000) / len(sine_chunked), 'ms')
print('')

print('####Testing Hard Distortion####')
start = time.perf_counter()
counter = 0
for counter in range(len(sine_chunked)):
    sine_chunked[counter] = harddistortiontest.apply(sine_chunked[counter])
    counter += 1
stop = time.perf_counter()
print('Success!')
print('Total Time: ', (stop - start) * 1000, 'ms')
print('Time per Chunk', ((stop - start) * 1000) / len(sine_chunked), 'ms')
print('')

print('####Testing Gate####')
start = time.perf_counter()
counter = 0
for counter in range(len(sine_chunked)):
    sine_chunked[counter] = gatetest.apply(sine_chunked[counter])
    counter += 1
stop = time.perf_counter()
print('Success!')
print('Total Time: ', (stop - start) * 1000, 'ms')
print('Time per Chunk', ((stop - start) * 1000) / len(sine_chunked), 'ms')
print('')

print('####Testing Lowcut FFT Filter####')
start = time.perf_counter()
counter = 0
for counter in range(len(sine_chunked)):
    sine_chunked[counter] = lowcuttest.apply(sine_chunked[counter])
    counter += 1
stop = time.perf_counter()
print('Success!')
print('Total Time: ', (stop - start) * 1000, 'ms')
print('Time per Chunk', ((stop - start) * 1000) / len(sine_chunked), 'ms')
print('')

print('####Testing Highcut FFT Filter####')
start = time.perf_counter()
counter = 0
for counter in range(len(sine_chunked)):
    sine_chunked[counter] = highcuttest.apply(sine_chunked[counter])
    counter += 1
stop = time.perf_counter()
print('Success!')
print('Total Time: ', (stop - start) * 1000, 'ms')
print('Time per Chunk', ((stop - start) * 1000) / len(sine_chunked), 'ms')
print('')

print('####Testing 3 Band EQ FFT Version####')
start = time.perf_counter()
counter = 0
for counter in range(len(sine_chunked)):
    sine_chunked[counter] = eq3bandffttest.apply(sine_chunked[counter])
    counter += 1
stop = time.perf_counter()
print('Success!')
print('Total Time: ', (stop - start) * 1000, 'ms')
print('Time per Chunk', ((stop - start) * 1000) / len(sine_chunked), 'ms')
print('')

print('####Testing Soft Clipper####')
start = time.perf_counter()
counter = 0
for counter in range(len(sine_chunked)):
    sine_chunked[counter] = softclippertest.apply(sine_chunked[counter])
    counter += 1
stop = time.perf_counter()
print('Success!')
print('Total Time: ', (stop - start) * 1000, 'ms')
print('Time per Chunk', ((stop - start) * 1000) / len(sine_chunked), 'ms')
print('')


sine_copy = CombineChunks(sine_chunked)


#pyplot.plot(sine_full)
#pyplot.plot(sine_copy)

#pyplot.show()

sys.exit()

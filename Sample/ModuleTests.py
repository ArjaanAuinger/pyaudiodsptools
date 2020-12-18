#!/usr/bin/env python3
# matplotlib for debugging
"""
Module Testing for pyaudiodsptools. Just run the script, make sure that the script is in the same folder
as the other scripts.
"""



import matplotlib.pyplot as pyplot

import config

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


from Generators import CreateSinewave, CreateSquarewave, CreateWhitenoise
from Utility import MakeChunks, CombineChunks, MixSignals, ConvertdBuTo16Bit, Convert16BitTodBu, ConvertdBVTo16Bit
from Utility import Convert16BitTodBV, Dither16BitTo8Bit, Dither32BitIntTo16BitInt, MonoWavToNumpy32BitFloat, InfodBV
from Utility import InfodBV16Bit, VolumeChange, MonoWavToNumpy16BitInt, Numpy16BitIntToMonoWav44kHz
from EffectCompressor import CreateCompressor
from EffectGate import CreateGate
from EffectDelay import CreateDelay
from _EffectReverb import CreateReverb
from EffectFFTFilter import CreateHighCutFilter, CreateLowCutFilter
from EffectEQ3BandFFT import CreateEQ3BandFFT
from EffectEQ3Band import CreateEQ3Band
from EffectLimiter import CreateLimiter
from EffectHardDistortion import CreateHardDistortion
from EffectTremolo import CreateTremolo
from EffectSaturator import CreateSaturator

print('####Creating Generators####')
sine_full = CreateSinewave(1000, 4096)
print('Sine Wave ok...')
square_full = CreateSquarewave(1000, 4096)
print('Square Wave ok...')
noise_full = CreateWhitenoise(4096)
print('White Noise ok..-')
print('')

# music_raw = MonoWavToNumpy32BitFloat('testmusic_mono.wav')
# music_raw = VolumeChange(music_raw,-0.0)
# music_raw = music_raw[0:88575]
# music_raw_copy = copy.deepcopy(music_raw)
# music_chunked = MakeChunks(music_raw_copy)

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
limitertest = CreateLimiter()
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

print('####Testing Limiter####')
start = time.perf_counter()
counter = 0
for counter in range(len(sine_chunked)):
    sine_chunked[counter] = limitertest.apply(sine_chunked[counter])
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

# music_copy = CombineChunks(music_chunked)
# music_copy = music_copy[512:]
sine_copy = CombineChunks(sine_chunked)

# pyplot.plot(music_raw)
# pyplot.plot(music_copy)
pyplot.plot(sine_full)
pyplot.plot(sine_copy)
pyplot.show()

# music_copy = music_copy*32767
# music_copy = music_copy.astype('int16')
# Numpy16BitIntToMonoWav44kHz("output.wav",music_copy)
print('wav created')

sys.exit()

#!/usr/bin/env python3
#matplotlib for debugging
import matplotlib.pyplot as pyplot

import os
import sys
import time
import timeit
import numpy
import math
import copy
import wave
import struct


from Generators import CreateSinewave, CreateSquarewave, CreateWhitenoise
from Utility import MakeChunks, CombineChunks, MixSignals, ConvertdBuTo16Bit, Convert16BitTodBu, ConvertdBVTo16Bit
from Utility import Convert16BitTodBV,Dither16BitTo8Bit, Dither32BitIntTo16BitInt, MonoWavToNumpy32BitFloat,InfodBV
from Utility import InfodBV16Bit, VolumeChange, MonoWavToNumpy16BitInt,Numpy16BitIntToMonoWav44kHz
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

#512 samples@44.1 kHz = 11.7ms = 0.00117s
#print(numpy.finfo('float64').max)# 1.79
#print(numpy.finfo('float64').min)# -1.79
#print(numpy.iinfo('int32').max)# 32767
#print(numpy.iinfo('int32').min)# -32768
#print(numpy.iinfo('int16').max)# 32767
#print(numpy.iinfo('int16').min)# -32768
#print(numpy.iinfo('int8').max)# 127
#print(numpy.iinfo('int8').min)# -128

#y = CreateWhitenoise(44100,512)
#y3 = CreateSquarewave(44100,1000,512)

sine_full = CreateSinewave(44100,10000,4096)
#sine_copy = copy.deepcopy(sine_full)

music_raw = MonoWavToNumpy32BitFloat('testmusic_mono.wav')
music_raw = VolumeChange(music_raw,-0.0)
#music_raw = music_raw[0:88575]

music_raw_copy = copy.deepcopy(music_raw)
sine_copy = copy.deepcopy(sine_full)

music_chunked = MakeChunks(music_raw_copy)
sine_chunked = MakeChunks(sine_copy)



harddistortion = CreateHardDistortion()
tremolotest = CreateTremolo()
delaytest = CreateDelay()
comptest = CreateCompressor()
limitertest = CreateLimiter()
saturatortest = CreateSaturator()
gatetest = CreateGate()
reverbtest = CreateReverb()
lowcut = CreateLowCutFilter(200)
eq3test = CreateEQ3Band(100,2,700,-4,8000,5)
eq3band = CreateEQ3BandFFT(100,2,700,-4,8000,5)


start = time.perf_counter()

counter = 0
for counter in range(len(music_chunked)):
    music_chunked[counter] = saturatortest.apply(music_chunked[counter])
    counter += 1

stop = time.perf_counter()

print('Total Time: ', (stop - start)*1000, 'ms')
print('Time per Chunk', ((stop - start)*1000)/len(music_chunked),'ms')

music_copy = CombineChunks(music_chunked)
#music_copy = music_copy[512:]
sine_copy = CombineChunks(sine_chunked)



pyplot.plot(music_raw)
pyplot.plot(music_copy)
#pyplot.plot(sine_full)
#pyplot.plot(sine_copy)
pyplot.show()

music_copy = music_copy*32767
music_copy = music_copy.astype('int16')
Numpy16BitIntToMonoWav44kHz("output.wav",music_copy)
print('wav created')








sys.exit()



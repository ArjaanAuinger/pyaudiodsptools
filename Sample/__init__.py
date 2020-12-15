import os
import sys
import numpy
import math
import copy
import wave
import struct

from .config import chunk_size, sampling_rate
from .Generators import CreateSinewave, CreateSquarewave, CreateWhitenoise
from .Utility import MakeChunks, CombineChunks, MixSignals, ConvertdBuTo16Bit, Convert16BitTodBu, ConvertdBVTo16Bit
from .Utility import Convert16BitTodBV,Dither16BitTo8Bit, Dither32BitIntTo16BitInt, MonoWavToNumpy32BitFloat,InfodBV
from .Utility import InfodBV16Bit, VolumeChange, MonoWavToNumpy16BitInt,Numpy16BitIntToMonoWav44kHz
from .EffectCompressor import CreateCompressor
from .EffectGate import CreateGate
from .EffectDelay import CreateDelay
from .EffectReverb import CreateReverb
from .EffectFFTFilter import CreateHighCutFilter, CreateLowCutFilter
from .EffectEQ3BandFFT import CreateEQ3BandFFT
from .EffectEQ3Band import CreateEQ3Band
from .EffectLimiter import CreateLimiter
from .EffectHardDistortion import CreateHardDistortion
from .EffectTremolo import CreateTremolo
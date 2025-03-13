import numpy

from . import config

try:
    import cupy
except ImportError:
    print('pyAudioDspTools: (Info) No cupy-cuda11x or cupy-cuda12x found. GPU rendering not possible.')
    config._gpu_available = False

from .Generators import CreateSinewave, CreateSquarewave, CreateWhitenoise
from .Utility import MakeChunks, CombineChunks, MixSignals, ConvertdBVTo16Bit
from .Utility import Convert16BitTodBV, Dither16BitTo8Bit, Dither32BitIntTo16BitInt, MonoWavToNumpyFloat, InfodBV
from .Utility import InfodBV16Bit, VolumeChange, MonoWavToNumpy16BitInt, NumpyFloatToWav
from .EffectCompressor import CreateCompressor
from .EffectSoftClipper import CreateSoftClipper
from .EffectSaturator import CreateSaturator
from .EffectGate import CreateGate
from .EffectDelay import CreateDelay
from .EffectFFTFilter import CreateHighCutFilter, CreateLowCutFilter
from .EffectEQ3BandFFT import CreateEQ3BandFFT
from .EffectEQ3Band import CreateEQ3Band
from .EffectHardDistortion import CreateHardDistortion
from .EffectTremolo import CreateTremolo

if config._gpu_available:
    from .EffectFFTFilterGPU import CreateHighCutFilterGPU, CreateLowCutFilterGPU
    from .EffectEQ3BandFFTGPU import CreateEQ3BandFFTGPU
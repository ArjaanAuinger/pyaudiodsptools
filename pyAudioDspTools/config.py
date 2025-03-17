"""Stores variables like config.chunk_size, also called buffer size by audio professionals, sampling rate and others.

Parameters
----------
config.sampling_rate : int
    Sets the global sampling rate of all classes/devices. Defaults to 44100 hertz, which is an audio standard.
config.chunk_size : int
    The number of samples in a chunk. Audio professionals might also call this buffer size, as this is the term
    used in a number of DAWs such as Ableton, Logic and Pro Tools.
config.use_gpu : bool
    If set to True, the FFT filter will use the GPU for processing. This is only possible if the cupy library is installed.

Notes
-----
* To set the sampling rate and config.chunk_size simply overwrite them in your script. Write this in the beginning of your
script: 'pyAudioDspTools.config.initialize(44100, 512)'

"""

import sys

# this is a pointer to the module object instance itself.
this = sys.modules[__name__]

# we can explicitly make assignments on it
this.sampling_rate = None
this.chunk_size = None
this.use_gpu = False
this._gpu_available = True # Will be overwritten if cupy is not available from __init__.py

def initialize(sampling_rate, chunk_size, use_gpu=False):
    #if (this.sampling_rate is None and this.chunk_size is None):
        # also in local function scope. no scope specifier like global is needed
    this.sampling_rate = sampling_rate
    this.chunk_size = chunk_size
    this.use_gpu = use_gpu
    #else:
        #msg = "config.py is already initialized to {0}."
        #raise RuntimeError(msg.format(this.sampling_rate))

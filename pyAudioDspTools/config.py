"""Stores variables like config.chunk_size, also called buffer size by audio professionals, sampling rate and others.
Has a default setting of 44100 Hz (44.1 kHz) and a chunk size of 512

Parameters
----------
config.sampling_rate : int
    Sets the global sampling rate of all classes/devices. Defaults to 44100 hertz, which is an audio standard.
config.chunk_size : int
    The number of samples in a chunk. Audio professionals might also call this buffer size, as this is the term
    used in a number of DAWs such as Ableton, Logic and Pro Tools.

Notes
-----
* To set the sampling rate and config.chunk_size simply overwrite them in your script. Write this in the beginning of your
script: 'pyAudioDspTools.config.sampling_rate = 48000' or 'pyAudioDspTools.config.chunk_size = 512'

"""

import sys

# this is a pointer to the module object instance itself.
this = sys.modules[__name__]

# we can explicitly make assignments on it
this.sampling_rate = None
this.chunk_size = None

def initialize(sampling_rate, chunk_size):
    if (this.sampling_rate is None and this.chunk_size is None):
        # also in local function scope. no scope specifier like global is needed
        this.sampling_rate = sampling_rate
        this.chunk_size = chunk_size
        # also the name remains free for local use
        db_name = "Locally scoped db_name variable. Doesn't do anything here."
    else:
        msg = "config.py is already initialized to {0}."
        raise RuntimeError(msg.format(this.sampling_rate))

#config.sampling_rate = 44100
#config.chunk_size = 512
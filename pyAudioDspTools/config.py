"""Stores variables like chunk_size, also called buffer size by audio professionals, sampling rate and others.
Has a default setting of 44100 Hz (44.1 kHz) and a chunk size of 512

Parameters
----------
sampling_rate : int
    Sets the global sampling rate of all classes/devices. Defaults to 44100 hertz, which is an audio standard.
chunk_size : int
    The number of samples in a chunk. Audio professionals might also call this buffer size, as this is the term
    used in a number of DAWs such as Ableton, Logic and Pro Tools.

Notes
-----
* To set the sampling rate and chunk_size simply overwrite them in your script. Write this in the beginning of your
script: 'pyAudioDspTools.sampling_rate = 48000' or 'pyAudioDspTools.chunk_size = 512'

"""



sampling_rate = 44100
chunk_size = 512
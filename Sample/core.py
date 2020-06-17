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
import wave
import struct

#512 samples@44.1 kHz = 11.7ms = 0.00117s
print(numpy.finfo('float64').max)# 1.79
print(numpy.finfo('float64').min)# -1.79
print(numpy.iinfo('int16').max)# 32767
print(numpy.iinfo('int16').min)# -32767

def CreateSinewave(sin_sample_rate, sin_frequency, sin_buffer_size):
    sin_time_array = numpy.arange(sin_buffer_size)
    sin_amplitude_array = numpy.int16(32767*numpy.sin(2 * numpy.pi * sin_frequency*sin_time_array/sin_sample_rate))
    return (sin_time_array, sin_amplitude_array)
def CreateSquarewave(square_sample_rate, square_frequency, square_buffer_size):
    square_time_array = numpy.arange(square_buffer_size)
    square_amplitude_array = numpy.int16(32767*numpy.sin(2 * numpy.pi * square_frequency * square_time_array / square_sample_rate))
    #print (square_amplitude_array)
    for sample in square_time_array:
            if square_amplitude_array[sample] > 0:
                square_amplitude_array[sample] = 32767
            if square_amplitude_array[sample] < 0:
                square_amplitude_array[sample] = -32767
    #print (square_amplitude_array)
    return (square_time_array, square_amplitude_array)
def CreateWhitenoise(buffer_size, sample_rate):
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

    whitenoise_amplitude_array = numpy.int16(32767*fftnoise(f))

    return whitenoise_time_array,whitenoise_amplitude_array


def ConvertdBuTo16Bit(float_array_input):
    float_array_input = numpy.where(float_array_input < 1.736, float_array_input, 1.736)
    float_array_input = numpy.where(float_array_input > -1.736, float_array_input, -1.736)
    float_array_output = numpy.int16(float_array_input * ((2 ** 15 - 1)/1.736))
    return float_array_output
def Convert16BitTodBu(float_array_input):
    float_array_output = numpy.float64((float_array_input/32767)*1.736)
    return float_array_output

def InfodBV(float_array_input):
    count = 0
    added_sample_value = 0.0
    average_sample_value = 0.0
    audio_array = copy.deepcopy(float_array_input)
    audio_array = numpy.where(audio_array > 0, audio_array, audio_array * (-1))
    average_sample_value = audio_array.sum()/audio_array.size
    amplitude = average_sample_value/1
    dBV = 20 * math.log10(amplitude)
    return dBV
def InfodBV16Bit(float_array_input):
    count = 0
    added_sample_value = 0.0
    average_sample_value = 0.0
    audio_array = copy.deepcopy(float_array_input)
    audio_array = numpy.where(audio_array > 0, audio_array, audio_array * (-1))
    average_sample_value = audio_array.sum()/audio_array.size
    amplitude = average_sample_value/32767
    dB16 = 20 * math.log10(amplitude)
    return dB16

def MonoWavToNumpy16BitInt(wav_file_path):
    wav_file = wave.open(wav_file_path)
    samples = wav_file.getnframes()
    audio = wav_file.readframes(samples)
    audio_as_numpy_array = np.frombuffer(audio, dtype=np.int16)
    return(audio_as_numpy_array)
def Numpy16BitIntToMonoWav(filename, data):
    """
    Write a numpy array as a WAV file

    Parameters
    ----------
    filename : string or open file handle
        Output wav file
    rate : int
        The sample rate (in samples/sec).
    data : ndarray
        A 1-D or 2-D numpy array of either integer or float data-type.

    Notes
    -----
    * The file can be an open file or a filename.

    * Writes a simple uncompressed WAV file.
    * The bits-per-sample will be determined by the data-type.
    * To write multiple-channels, use a 2-D array of shape
      (Nsamples, Nchannels).

    """
    if hasattr(filename,'write'):
        fid = filename
    else:
        fid = open(filename, 'wb')

    try:
        dkind = data.dtype.kind
        if not (dkind == 'i' or dkind == 'f' or (dkind == 'u' and data.dtype.itemsize == 1)):
            raise ValueError("Unsupported data type '%s'" % data.dtype)

        fid.write(b'RIFF')
        fid.write(b'\x00\x00\x00\x00')
        fid.write(b'WAVE')
        # fmt chunk
        fid.write(b'fmt ')
        if dkind == 'f':
            comp = 3
        else:
            comp = 1
        if data.ndim == 1:
            noc = 1
        else:
            noc = data.shape[1]
        bits = data.dtype.itemsize * 8
        sbytes = 44100*(bits // 8)*noc
        ba = noc * (bits // 8)
        fid.write(struct.pack('<ihHIIHH', 16, comp, noc, 44100, sbytes, ba, bits))
        # data chunk
        fid.write(b'data')
        fid.write(struct.pack('<i', data.nbytes))
        if data.dtype.byteorder == '>' or (data.dtype.byteorder == '=' and sys.byteorder == 'big'):
            data = data.byteswap()
        _array_tofile(fid, data)

        # Determine file size and place it in correct
        #  position at start of the file.
        size = fid.tell()
        fid.seek(4)
        fid.write(struct.pack('<i', size-8))

    finally:
        if not hasattr(filename,'write'):
            fid.close()
        else:
            fid.seek(0)
    return

if sys.version_info[0] >= 3:
    def _array_tofile(fid, data):
        # ravel gives a c-contiguous buffer
        fid.write(data.ravel().view('b').data)
else:
    def _array_tofile(fid, data):
        fid.write(data.tostring())




x,y = CreateWhitenoise(512,44100)

x2,y2 = CreateSinewave(44100,1000,88200)

x3,y3 = CreateSquarewave(44100,1000,512)



start = timeit.default_timer()
Numpy16BitIntToMonoWav('test.wav', y2)
stop = timeit.default_timer()
print('Time: ', (stop - start)*1000, 'ms')



pyplot.plot(x, y)
pyplot.plot(x2,y2)
pyplot.plot(x3,y3)

pyplot.show()



sys.exit()



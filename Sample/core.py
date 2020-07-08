#!/usr/bin/env python3
import os
import sys
import time
import timeit
import numpy
import math
import matplotlib.pyplot as pyplot
#import scipy
import copy
import wave
import struct

#512 samples@44.1 kHz = 11.7ms = 0.00117s
#print(numpy.finfo('float64').max)# 1.79
#print(numpy.finfo('float64').min)# -1.79
#print(numpy.iinfo('int32').max)# 32767
#print(numpy.iinfo('int32').min)# -32768
#print(numpy.iinfo('int16').max)# 32767
#print(numpy.iinfo('int16').min)# -32768
#print(numpy.iinfo('int8').max)# 127
#print(numpy.iinfo('int8').min)# -128

def CreateSinewave(sin_sample_rate, sin_frequency, sin_buffer_size):
    sin_time_array = numpy.arange(sin_buffer_size)
    sin_amplitude_array = numpy.int16(32767*numpy.sin(2 * numpy.pi * sin_frequency*sin_time_array/sin_sample_rate))
    return (sin_amplitude_array)

def CreateSinewave32Bit(sin_sample_rate, sin_frequency, sin_buffer_size):
    sin_time_array = numpy.arange(sin_buffer_size)
    sin_amplitude_array = numpy.int32(2147483647*numpy.sin(2 * numpy.pi * sin_frequency*sin_time_array/sin_sample_rate))
    return (sin_amplitude_array)

def CreateSquarewave(square_sample_rate, square_frequency, square_buffer_size):
    square_time_array = numpy.arange(square_buffer_size)
    square_amplitude_array = numpy.int16(32767*numpy.sin(2 * numpy.pi * square_frequency * square_time_array / square_sample_rate))
    square_amplitude_array = numpy.where(square_amplitude_array>0,32767,-32767)
    return (square_amplitude_array)

def CreateWhitenoise(sample_rate,buffer_size):
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

def Convert16BitTodBu(int_array_input):
    float_array_output = numpy.float32((int_array_input/32767)*1.736)
    return float_array_output

def ConvertdBVTo16Bit(float_array_input):
    float_array_input = numpy.where(float_array_input < 1, float_array_input, 1)
    float_array_input = numpy.where(float_array_input > -1, float_array_input, -1)
    float_array_output = numpy.int16(float_array_input * (2 ** 15 - 1))
    return float_array_output

def Convert16BitTodBV(int_array_input):
    float_array_output = numpy.float32(int_array_input/32767)
    return float_array_output

def Dither16BitTo8Bit(int_array_input):
    rectangular_dither_array = numpy.random.randint(-1, 1, size=int_array_input.size)
    int_array_dithered = numpy.around(int_array_input / 256, decimals=0).astype('int16')

    int_array_dithered = numpy.add(int_array_dithered, rectangular_dither_array)
    int_array_dithered = numpy.clip(int_array_dithered, a_min=-127, a_max=127)
    int_array_dithered.astype('int8')
    # int_array_output = (int_array_dithered*256).astype('int16')
    return int_array_dithered

def Dither32BitTo16Bit(int_array_input):
    rectangular_dither_array = numpy.random.randint(-1, 1, size=int_array_input.size)
    int_array_dithered = numpy.around(int_array_input / 65535, decimals=0).astype('int32')

    int_array_dithered = numpy.add(int_array_dithered, rectangular_dither_array)
    int_array_dithered = numpy.clip(int_array_dithered, a_min=-32767, a_max=32767)
    int_array_dithered = int_array_dithered.astype('int16')
    # int_array_output = (int_array_dithered*65535).astype('int32')
    return int_array_dithered

#BSD Licence
#Not Tested!
def Import24BitWavTo16Bit(wav_file,data):
    if sampwidth != 3:
        print("wav_file is not 24-Bit! Cannot perform operation.")
        return
    else:
        a = numpy.empty((num_samples, nchannels, 4), dtype=numpy.uint8)
        raw_bytes = numpy.fromstring(data, dtype=numpy.uint8)
        a[:, :, :sampwidth] = raw_bytes.reshape(-1, nchannels, sampwidth)
        a[:, :, sampwidth:] = (a[:, :, sampwidth - 1:sampwidth] >> 7) * 255
        result = a.view('<i4').reshape(a.shape[:-1])
        return result

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

def InfodBV16Bit(int_array_input):
    count = 0
    added_sample_value = 0.0
    average_sample_value = 0.0
    audio_array = copy.deepcopy(int_array_input)
    audio_array = numpy.where(audio_array > 0, audio_array, audio_array * (-1))
    average_sample_value = audio_array.sum()/audio_array.size
    amplitude = average_sample_value/32767
    dB16 = 20 * math.log10(amplitude)
    return dB16

def XaxisForMatplotlib(any_array_input):
    any_array_output = numpy.arange(any_array_input.size)
    return any_array_output


def VolumeChange16Bit(int_array_input, gain_change_in_db):
    float_array_input = numpy.float32(int_array_input/32767)
    float_array_input = (10 ** (gain_change_in_db/20))*float_array_input
    float_array_input = numpy.clip(float_array_input, -1.0, 1.0)
    int_array_output = numpy.int16(float_array_input*32767)
    return int_array_output


def MonoWavToNumpy16BitInt(wav_file_path):
    wav_file = wave.open(wav_file_path)
    samples = wav_file.getnframes()
    audio = wav_file.readframes(samples)
    audio_as_numpy_array = numpy.frombuffer(audio, dtype=numpy.int16)
    return(audio_as_numpy_array)

def Numpy16BitIntToMonoWav44kHz(filename, data):
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




def InvertSignal(int_array_input):
    int_array_output = numpy.invert(int_array_input)
    return(int_array_output)

def EffectCompressor(int_array_input, threshold=-10.0, ratio=4.0, attack=5.0, release=50.0):
    db_array_input = numpy.float32(10 ** (threshold / 20))
    db_array_int = numpy.int16(db_array_input*32767)
    print(db_array_int)

class EffectLimiter:
    def __init__(self, threshold_in_db, attack_coeff=0.9, release_coeff=0.992, delay=10):
        self.delay_index = 0
        self.envelope = 0
        self.gain = 1
        self.delay = delay
        self.delay_line = numpy.zeros(delay)
        self.release_coeff = release_coeff
        self.attack_coeff = attack_coeff
        self.threshold = numpy.int16((10 ** (threshold_in_db/20))*32767)
        print(self.threshold)

    def limit(self, signal):
        for idx, sample in enumerate(signal):
            self.delay_line[self.delay_index] = sample
            self.delay_index = (self.delay_index + 1) % self.delay

            # calculate an envelope of the signal
            self.envelope  = max(abs(sample), self.envelope*self.release_coeff)

            if self.envelope > self.threshold:
                target_gain = self.threshold / self.envelope
            else:
                target_gain = 1.0

            # have self.gain go towards a desired limiter gain
            self.gain = ( self.gain*self.attack_coeff +
                          target_gain*(1-self.attack_coeff) )

            # limit the delayed signal
            signal[idx] = self.delay_line[self.delay_index] * self.gain
        return signal


class EffectTremolo:
    def __init__(self,tremolo_depth,lfo_in_hertz=4.5):
        self.sin_sample_rate = 44100
        self.sin_time_array = numpy.arange(numpy.float32(self.sin_sample_rate/lfo_in_hertz))
        self.sin_lfo = numpy.float32((((numpy.sin(2 * numpy.pi * lfo_in_hertz*self.sin_time_array/self.sin_sample_rate)
                                        /2)+0.5)*tremolo_depth)+(1-tremolo_depth))
        self.lfo_length = len(self.sin_lfo)
        self.sin_lfo_copy = copy.deepcopy(self.sin_lfo)

    def tremolo(self,int_array_input):
        current_input_lenght = len(int_array_input)
        while len(self.sin_lfo_copy) < len(int_array_input):
            self.sin_lfo_copy = numpy.append(self.sin_lfo_copy,self.sin_lfo)
        self.sin_lfo_chunk = self.sin_lfo_copy[:current_input_lenght]
        self.sin_lfo_copy = self.sin_lfo_copy[-(len(self.sin_lfo_copy)-current_input_lenght):]

        print(len(self.sin_lfo_chunk))
        print(current_input_lenght)
        numpy.multiply(int_array_input,self.sin_lfo_chunk, out=int_array_input, dtype='float32', casting='unsafe')
        int_array_output = int_array_input.astype('int16')
        return int_array_output

    def lfo_reset(self):
        self.sin_lfo_copy = copy.deepcopy(self.sin_lfo)

def EffectHardDistortion(int_array_input):
    hard_limit = 32767
    linear_limit = 32000
    clip_limit = linear_limit + int(numpy.pi / 2 * (hard_limit - linear_limit))
    sign = copy.deepcopy(int_array_input)
    sign = numpy.where(int_array_input>=0,1,-1)
    amplitude = numpy.absolute(int_array_input)
    amplitude = numpy.where(amplitude <= linear_limit, amplitude,hard_limit*sign)
    scale = hard_limit - linear_limit
    compression = scale * numpy.sin(numpy.float32(amplitude - linear_limit) / scale)
    output = numpy.int16((linear_limit + numpy.int16(compression)) * sign)
    return output







#y = CreateWhitenoise(44100,512)
#y3 = CreateSquarewave(44100,1000,512)
sine_full = CreateSinewave(44100,100,8820)
music = MonoWavToNumpy16BitInt('testmusic_mono.wav')
music_copy = copy.deepcopy(music)
#sine,sine2,sine3,sine4 = numpy.split(sine_full,4)

tremolo = EffectTremolo(0.5)

#Numpy16BitIntToMonoWav('test.wav',y22)
start = timeit.default_timer()
wave = tremolo.tremolo(sine_full)
#limiter.limit(sine2)
#limiter.limit(sine3)
#limiter.limit(sine4)
#sine_add = numpy.append(sine,sine2)
#sine_add = numpy.append(sine_add,sine3)
#sine_add = numpy.append(sine_add,sine4)
#Numpy16BitIntToMonoWav44kHz("output.wav",sine)
stop = timeit.default_timer()
print('Time: ', (stop - start)*1000, 'ms')

#pyplot.plot(XaxisForMatplotlib(sine),sine)
#pyplot.plot(XaxisForMatplotlib(y3),y3)
#pyplot.plot(XaxisForMatplotlib(sine), sine)
pyplot.plot(XaxisForMatplotlib(wave),wave)
#pyplot.plot(XaxisForMatplotlib(y6),y6)


pyplot.show()



sys.exit()



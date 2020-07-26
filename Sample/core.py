#!/usr/bin/env python3
import os
import sys
import time
import timeit
import numpy
import math

import operator
from numpy.polynomial.polynomial import polyval as npp_polyval

import matplotlib.pyplot as pyplot
#import scipy
#from scipy.fftpack import fftfreq, irfft, rfft
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

#Still need to add clipping
def MixSignals(*args):
    for signal in args:
        try:
            mixed_signal = mixed_signal + signal
        except:
            raise Exception("Something went wrong. Make sure, that the Numpy arrays are equal in length.")
    return mixed_signal

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
        numpy.multiply(int_array_input,self.sin_lfo_chunk, out=int_array_input, dtype='float32', casting='unsafe')
        int_array_output = int_array_input.astype('int16')
        return int_array_output

    def lfo_reset(self):
        self.sin_lfo_copy = copy.deepcopy(self.sin_lfo)

#class Effect3BandEQ:
    #def __init__(selfhi_gain=0,mid_gain=0,low_gain=0):

def EQ(int_array_input):
    int_array_input_length = numpy.int16(len(int_array_input))

    fs = 1000
    t = numpy.linspace(0, 1000 / fs, 1000, endpoint=False)
    f = 3.0  # Frequency in Hz
    A = 100.0  # Amplitude in Unit
    s = A * numpy.sin(2 * numpy.pi * f * t)  # Signal
    dt = 1 / fs

    W = fftfreq(s.size, d=dt)
    f_signal = rfft(s)

    cut_f_signal = f_signal.copy()
    #cut_f_signal[(numpy.abs(W) > 3)] = 0  # cut signal above 3Hz

    cs = irfft(cut_f_signal)

    #int_array_input = (int_array_input.astype('float32')/32768)+1
    #x = np.arange(0, 10, 10 / len(int_array_input))
    #freqs = np.fft.fftfreq(len(x), .01)
    #n = int_array_input.size
    #int_array_input = numpy.where(int_array_input<=0,1,int_array_input)
    #int_array_input = numpy.fft.fftfreq(n,d=44100/1)
    #print (int_array_input)
    #int_array_input_fft = numpy.fft.fft(int_array_input)[0:int_array_input_length]
    #int_array_input_fft = numpy.absolute(int_array_input_fft)
    freq = numpy.int16(44100 * numpy.arange(int_array_input_length) / (int_array_input_length*2))
    #print(freq)# frequency vector
    #int_array_input_fft = numpy.fft.ifft(int_array_input_fft)
    return cs,freq

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


def buttap(N):
    """Return (z,p,k) for analog prototype of Nth-order Butterworth filter.
    The filter will have an angular (e.g., rad/s) cutoff frequency of 1.
    See Also
    --------
    butter : Filter design function using this prototype
    """
    if abs(int(N)) != N:
        raise ValueError("Filter order must be a nonnegative integer")
    z = numpy.array([])
    m = numpy.arange(-N+1, N, 2)
    # Middle value is 0 to ensure an exactly real pole
    p = -numpy.exp(1j * numpy.pi * m / (2 * N))
    k = 1
    return z, p, k

def buttord(wp, ws, gpass, gstop, analog=False, fs=None):
    """Butterworth filter order selection.
    Return the order of the lowest order digital or analog Butterworth filter
    that loses no more than `gpass` dB in the passband and has at least
    `gstop` dB attenuation in the stopband.
    Parameters
    ----------
    wp, ws : float
        Passband and stopband edge frequencies.
        For digital filters, these are in the same units as `fs`. By default,
        `fs` is 2 half-cycles/sample, so these are normalized from 0 to 1,
        where 1 is the Nyquist frequency. (`wp` and `ws` are thus in
        half-cycles / sample.) For example:
            - Lowpass:   wp = 0.2,          ws = 0.3
            - Highpass:  wp = 0.3,          ws = 0.2
            - Bandpass:  wp = [0.2, 0.5],   ws = [0.1, 0.6]
            - Bandstop:  wp = [0.1, 0.6],   ws = [0.2, 0.5]
        For analog filters, `wp` and `ws` are angular frequencies (e.g., rad/s).
    gpass : float
        The maximum loss in the passband (dB).
    gstop : float
        The minimum attenuation in the stopband (dB).
    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned.
    fs : float, optional
        The sampling frequency of the digital system.
        .. versionadded:: 1.2.0
    Returns
    -------
    ord : int
        The lowest order for a Butterworth filter which meets specs.
    wn : ndarray or float
        The Butterworth natural frequency (i.e. the "3dB frequency"). Should
        be used with `butter` to give filter results. If `fs` is specified,
        this is in the same units, and `fs` must also be passed to `butter`.
    See Also
    --------
    butter : Filter design using order and critical points
    cheb1ord : Find order and critical points from passband and stopband spec
    cheb2ord, ellipord
    iirfilter : General filter design using order and critical frequencies
    iirdesign : General filter design using passband and stopband spec
    Examples
    --------
    Design an analog bandpass filter with passband within 3 dB from 20 to
    50 rad/s, while rejecting at least -40 dB below 14 and above 60 rad/s.
    Plot its frequency response, showing the passband and stopband
    constraints in gray.
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> N, Wn = signal.buttord([20, 50], [14, 60], 3, 40, True)
    >>> b, a = signal.butter(N, Wn, 'band', True)
    >>> w, h = signal.freqs(b, a, np.logspace(1, 2, 500))
    >>> plt.semilogx(w, 20 * np.log10(abs(h)))
    >>> plt.title('Butterworth bandpass filter fit to constraints')
    >>> plt.xlabel('Frequency [radians / second]')
    >>> plt.ylabel('Amplitude [dB]')
    >>> plt.grid(which='both', axis='both')
    >>> plt.fill([1,  14,  14,   1], [-40, -40, 99, 99], '0.9', lw=0) # stop
    >>> plt.fill([20, 20,  50,  50], [-99, -3, -3, -99], '0.9', lw=0) # pass
    >>> plt.fill([60, 60, 1e9, 1e9], [99, -40, -40, 99], '0.9', lw=0) # stop
    >>> plt.axis([10, 100, -60, 3])
    >>> plt.show()
    """

    _validate_gpass_gstop(gpass, gstop)

    wp = atleast_1d(wp)
    ws = atleast_1d(ws)
    if fs is not None:
        if analog:
            raise ValueError("fs cannot be specified for an analog filter")
        wp = 2*wp/fs
        ws = 2*ws/fs

    filter_type = 2 * (len(wp) - 1)
    filter_type += 1
    if wp[0] >= ws[0]:
        filter_type += 1

    # Pre-warp frequencies for digital filter design
    if not analog:
        passb = tan(pi * wp / 2.0)
        stopb = tan(pi * ws / 2.0)
    else:
        passb = wp * 1.0
        stopb = ws * 1.0

    if filter_type == 1:            # low
        nat = stopb / passb
    elif filter_type == 2:          # high
        nat = passb / stopb
    elif filter_type == 3:          # stop
        wp0 = optimize.fminbound(band_stop_obj, passb[0], stopb[0] - 1e-12,
                                 args=(0, passb, stopb, gpass, gstop,
                                       'butter'),
                                 disp=0)
        passb[0] = wp0
        wp1 = optimize.fminbound(band_stop_obj, stopb[1] + 1e-12, passb[1],
                                 args=(1, passb, stopb, gpass, gstop,
                                       'butter'),
                                 disp=0)
        passb[1] = wp1
        nat = ((stopb * (passb[0] - passb[1])) /
               (stopb ** 2 - passb[0] * passb[1]))
    elif filter_type == 4:          # pass
        nat = ((stopb ** 2 - passb[0] * passb[1]) /
               (stopb * (passb[0] - passb[1])))

    nat = min(abs(nat))

    GSTOP = 10 ** (0.1 * abs(gstop))
    GPASS = 10 ** (0.1 * abs(gpass))
    ord = int(ceil(log10((GSTOP - 1.0) / (GPASS - 1.0)) / (2 * log10(nat))))

    # Find the Butterworth natural frequency WN (or the "3dB" frequency")
    # to give exactly gpass at passb.
    try:
        W0 = (GPASS - 1.0) ** (-1.0 / (2.0 * ord))
    except ZeroDivisionError:
        W0 = 1.0
        print("Warning, order is zero...check input parameters.")

    # now convert this frequency back from lowpass prototype
    # to the original analog filter

    if filter_type == 1:  # low
        WN = W0 * passb
    elif filter_type == 2:  # high
        WN = passb / W0
    elif filter_type == 3:  # stop
        WN = numpy.zeros(2, float)
        discr = sqrt((passb[1] - passb[0]) ** 2 +
                     4 * W0 ** 2 * passb[0] * passb[1])
        WN[0] = ((passb[1] - passb[0]) + discr) / (2 * W0)
        WN[1] = ((passb[1] - passb[0]) - discr) / (2 * W0)
        WN = numpy.sort(abs(WN))
    elif filter_type == 4:  # pass
        W0 = numpy.array([-W0, W0], float)
        WN = (-W0 * (passb[1] - passb[0]) / 2.0 +
              sqrt(W0 ** 2 / 4.0 * (passb[1] - passb[0]) ** 2 +
                   passb[0] * passb[1]))
        WN = numpy.sort(abs(WN))
    else:
        raise ValueError("Bad type: %s" % filter_type)

    if not analog:
        wn = (2.0 / pi) * arctan(WN)
    else:
        wn = WN

    if len(wn) == 1:
        wn = wn[0]

    if fs is not None:
        wn = wn*fs/2

    return ord, wn

def _relative_degree(z, p):
    """
    Return relative degree of transfer function from zeros and poles
    """
    degree = len(p) - len(z)
    if degree < 0:
        raise ValueError("Improper transfer function. "
                         "Must have at least as many poles as zeros.")
    else:
        return degree

def lp2bp_zpk(z, p, k, wo=1.0, bw=1.0):
    r"""
    Transform a lowpass filter prototype to a bandpass filter.
    Return an analog band-pass filter with center frequency `wo` and
    bandwidth `bw` from an analog low-pass filter prototype with unity
    cutoff frequency, using zeros, poles, and gain ('zpk') representation.
    Parameters
    ----------
    z : array_like
        Zeros of the analog filter transfer function.
    p : array_like
        Poles of the analog filter transfer function.
    k : float
        System gain of the analog filter transfer function.
    wo : float
        Desired passband center, as angular frequency (e.g., rad/s).
        Defaults to no change.
    bw : float
        Desired passband width, as angular frequency (e.g., rad/s).
        Defaults to 1.
    Returns
    -------
    z : ndarray
        Zeros of the transformed band-pass filter transfer function.
    p : ndarray
        Poles of the transformed band-pass filter transfer function.
    k : float
        System gain of the transformed band-pass filter.
    See Also
    --------
    lp2lp_zpk, lp2hp_zpk, lp2bs_zpk, bilinear
    lp2bp
    Notes
    -----
    This is derived from the s-plane substitution
    .. math:: s \rightarrow \frac{s^2 + {\omega_0}^2}{s \cdot \mathrm{BW}}
    This is the "wideband" transformation, producing a passband with
    geometric (log frequency) symmetry about `wo`.
    .. versionadded:: 1.1.0
    """
    z = numpy.atleast_1d(z)
    p = numpy.atleast_1d(p)
    wo = float(wo)
    bw = float(bw)

    degree = _relative_degree(z, p)

    # Scale poles and zeros to desired bandwidth
    z_lp = z * bw/2
    p_lp = p * bw/2

    # Square root needs to produce complex result, not NaN
    z_lp = z_lp.astype(complex)
    p_lp = p_lp.astype(complex)

    # Duplicate poles and zeros and shift from baseband to +wo and -wo
    z_bp = numpy.concatenate((z_lp + numpy.sqrt(z_lp**2 - wo**2),
                        z_lp - numpy.sqrt(z_lp**2 - wo**2)))
    p_bp = numpy.concatenate((p_lp + numpy.sqrt(p_lp**2 - wo**2),
                        p_lp - numpy.sqrt(p_lp**2 - wo**2)))

    # Move degree zeros to origin, leaving degree zeros at infinity for BPF
    z_bp = numpy.append(z_bp, numpy.zeros(degree))

    # Cancel out gain change from frequency scaling
    k_bp = k * bw**degree

    return z_bp, p_bp, k_bp
def bilinear_zpk(z, p, k, fs):
    r"""
    Return a digital IIR filter from an analog one using a bilinear transform.
    Transform a set of poles and zeros from the analog s-plane to the digital
    z-plane using Tustin's method, which substitutes ``(z-1) / (z+1)`` for
    ``s``, maintaining the shape of the frequency response.
    Parameters
    ----------
    z : array_like
        Zeros of the analog filter transfer function.
    p : array_like
        Poles of the analog filter transfer function.
    k : float
        System gain of the analog filter transfer function.
    fs : float
        Sample rate, as ordinary frequency (e.g., hertz). No prewarping is
        done in this function.
    Returns
    -------
    z : ndarray
        Zeros of the transformed digital filter transfer function.
    p : ndarray
        Poles of the transformed digital filter transfer function.
    k : float
        System gain of the transformed digital filter.
    See Also
    --------
    lp2lp_zpk, lp2hp_zpk, lp2bp_zpk, lp2bs_zpk
    bilinear
    Notes
    -----
    .. versionadded:: 1.1.0
    Examples
    --------
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> fs = 100
    >>> bf = 2 * np.pi * np.array([7, 13])
    >>> filts = signal.lti(*signal.butter(4, bf, btype='bandpass', analog=True, output='zpk'))
    >>> filtz = signal.lti(*signal.bilinear_zpk(filts.zeros, filts.poles, filts.gain, fs))
    >>> wz, hz = signal.freqz_zpk(filtz.zeros, filtz.poles, filtz.gain)
    >>> ws, hs = signal.freqs_zpk(filts.zeros, filts.poles, filts.gain, worN=fs*wz)
    >>> plt.semilogx(wz*fs/(2*np.pi), 20*np.log10(np.abs(hz).clip(1e-15)), label=r'$|H(j \omega)|$')
    >>> plt.semilogx(wz*fs/(2*np.pi), 20*np.log10(np.abs(hs).clip(1e-15)), label=r'$|H_z(e^{j \omega})|$')
    >>> plt.legend()
    >>> plt.xlabel('Frequency [Hz]')
    >>> plt.ylabel('Magnitude [dB]')
    >>> plt.grid()
    """
    z = numpy.atleast_1d(z)
    p = numpy.atleast_1d(p)

    degree = _relative_degree(z, p)

    fs2 = 2.0*fs

    # Bilinear transform the poles and zeros
    z_z = (fs2 + z) / (fs2 - z)
    p_z = (fs2 + p) / (fs2 - p)

    # Any zeros that were at infinity get moved to the Nyquist frequency
    z_z = numpy.append(z_z, -numpy.ones(degree))

    # Compensate for gain change
    k_z = k * numpy.real(numpy.prod(fs2 - z) / numpy.prod(fs2 - p))

    return z_z, p_z, k_z
def _cplxreal(z, tol=None):
    """
    Split into complex and real parts, combining conjugate pairs.
    The 1-D input vector `z` is split up into its complex (`zc`) and real (`zr`)
    elements. Every complex element must be part of a complex-conjugate pair,
    which are combined into a single number (with positive imaginary part) in
    the output. Two complex numbers are considered a conjugate pair if their
    real and imaginary parts differ in magnitude by less than ``tol * abs(z)``.
    Parameters
    ----------
    z : array_like
        Vector of complex numbers to be sorted and split
    tol : float, optional
        Relative tolerance for testing realness and conjugate equality.
        Default is ``100 * spacing(1)`` of `z`'s data type (i.e., 2e-14 for
        float64)
    Returns
    -------
    zc : ndarray
        Complex elements of `z`, with each pair represented by a single value
        having positive imaginary part, sorted first by real part, and then
        by magnitude of imaginary part. The pairs are averaged when combined
        to reduce error.
    zr : ndarray
        Real elements of `z` (those having imaginary part less than
        `tol` times their magnitude), sorted by value.
    Raises
    ------
    ValueError
        If there are any complex numbers in `z` for which a conjugate
        cannot be found.
    See Also
    --------
    _cplxpair
    Examples
    --------
    >>> a = [4, 3, 1, 2-2j, 2+2j, 2-1j, 2+1j, 2-1j, 2+1j, 1+1j, 1-1j]
    >>> zc, zr = _cplxreal(a)
    >>> print(zc)
    [ 1.+1.j  2.+1.j  2.+1.j  2.+2.j]
    >>> print(zr)
    [ 1.  3.  4.]
    """

    z = numpy.atleast_1d(z)
    if z.size == 0:
        return z, z
    elif z.ndim != 1:
        raise ValueError('_cplxreal only accepts 1-D input')

    if tol is None:
        # Get tolerance from dtype of input
        tol = 100 * numpy.finfo((1.0 * z).dtype).eps

    # Sort by real part, magnitude of imaginary part (speed up further sorting)
    z = z[numpy.lexsort((abs(z.imag), z.real))]

    # Split reals from conjugate pairs
    real_indices = abs(z.imag) <= tol * abs(z)
    zr = z[real_indices].real

    if len(zr) == len(z):
        # Input is entirely real
        return numpy.array([]), zr

    # Split positive and negative halves of conjugates
    z = z[~real_indices]
    zp = z[z.imag > 0]
    zn = z[z.imag < 0]

    if len(zp) != len(zn):
        raise ValueError('Array contains complex value with no matching '
                         'conjugate.')

    # Find runs of (approximately) the same real part
    same_real = numpy.diff(zp.real) <= tol * abs(zp[:-1])
    diffs = numpy.diff(numpy.concatenate(([0], same_real, [0])))
    run_starts = numpy.nonzero(diffs > 0)[0]
    run_stops = numpy.nonzero(diffs < 0)[0]

    # Sort each run by their imaginary parts
    for i in range(len(run_starts)):
        start = run_starts[i]
        stop = run_stops[i] + 1
        for chunk in (zp[start:stop], zn[start:stop]):
            chunk[...] = chunk[np.lexsort([abs(chunk.imag)])]

    # Check that negatives match positives
    if any(abs(zp - zn.conj()) > tol * abs(zn)):
        raise ValueError('Array contains complex value with no matching '
                         'conjugate.')

    # Average out numerical inaccuracy in real vs imag parts of pairs
    zc = (zp + zn.conj()) / 2

    return zc, zr

def _nearest_real_complex_idx(fro, to, which):
    """Get the next closest real or complex element based on distance"""
    assert which in ('real', 'complex')
    order = numpy.argsort(numpy.abs(fro - to))
    mask = numpy.isreal(fro[order])
    if which == 'complex':
        mask = ~mask
    return order[numpy.nonzero(mask)[0][0]]

def zpk2sos(z, p, k, pairing='nearest'):
    """
    Return second-order sections from zeros, poles, and gain of a system
    Parameters
    ----------
    z : array_like
        Zeros of the transfer function.
    p : array_like
        Poles of the transfer function.
    k : float
        System gain.
    pairing : {'nearest', 'keep_odd'}, optional
        The method to use to combine pairs of poles and zeros into sections.
        See Notes below.
    Returns
    -------
    sos : ndarray
        Array of second-order filter coefficients, with shape
        ``(n_sections, 6)``. See `sosfilt` for the SOS filter format
        specification.
    See Also
    --------
    sosfilt
    Notes
    -----
    The algorithm used to convert ZPK to SOS format is designed to
    minimize errors due to numerical precision issues. The pairing
    algorithm attempts to minimize the peak gain of each biquadratic
    section. This is done by pairing poles with the nearest zeros, starting
    with the poles closest to the unit circle.
    *Algorithms*
    The current algorithms are designed specifically for use with digital
    filters. (The output coefficients are not correct for analog filters.)
    The steps in the ``pairing='nearest'`` and ``pairing='keep_odd'``
    algorithms are mostly shared. The ``nearest`` algorithm attempts to
    minimize the peak gain, while ``'keep_odd'`` minimizes peak gain under
    the constraint that odd-order systems should retain one section
    as first order. The algorithm steps and are as follows:
    As a pre-processing step, add poles or zeros to the origin as
    necessary to obtain the same number of poles and zeros for pairing.
    If ``pairing == 'nearest'`` and there are an odd number of poles,
    add an additional pole and a zero at the origin.
    The following steps are then iterated over until no more poles or
    zeros remain:
    1. Take the (next remaining) pole (complex or real) closest to the
       unit circle to begin a new filter section.
    2. If the pole is real and there are no other remaining real poles [#]_,
       add the closest real zero to the section and leave it as a first
       order section. Note that after this step we are guaranteed to be
       left with an even number of real poles, complex poles, real zeros,
       and complex zeros for subsequent pairing iterations.
    3. Else:
        1. If the pole is complex and the zero is the only remaining real
           zero*, then pair the pole with the *next* closest zero
           (guaranteed to be complex). This is necessary to ensure that
           there will be a real zero remaining to eventually create a
           first-order section (thus keeping the odd order).
        2. Else pair the pole with the closest remaining zero (complex or
           real).
        3. Proceed to complete the second-order section by adding another
           pole and zero to the current pole and zero in the section:
            1. If the current pole and zero are both complex, add their
               conjugates.
            2. Else if the pole is complex and the zero is real, add the
               conjugate pole and the next closest real zero.
            3. Else if the pole is real and the zero is complex, add the
               conjugate zero and the real pole closest to those zeros.
            4. Else (we must have a real pole and real zero) add the next
               real pole closest to the unit circle, and then add the real
               zero closest to that pole.
    .. [#] This conditional can only be met for specific odd-order inputs
           with the ``pairing == 'keep_odd'`` method.
    .. versionadded:: 0.16.0
    Examples
    --------
    Design a 6th order low-pass elliptic digital filter for a system with a
    sampling rate of 8000 Hz that has a pass-band corner frequency of
    1000 Hz. The ripple in the pass-band should not exceed 0.087 dB, and
    the attenuation in the stop-band should be at least 90 dB.
    In the following call to `signal.ellip`, we could use ``output='sos'``,
    but for this example, we'll use ``output='zpk'``, and then convert to SOS
    format with `zpk2sos`:
    >>> from scipy import signal
    >>> z, p, k = signal.ellip(6, 0.087, 90, 1000/(0.5*8000), output='zpk')
    Now convert to SOS format.
    >>> sos = signal.zpk2sos(z, p, k)
    The coefficients of the numerators of the sections:
    >>> sos[:, :3]
    array([[ 0.0014154 ,  0.00248707,  0.0014154 ],
           [ 1.        ,  0.72965193,  1.        ],
           [ 1.        ,  0.17594966,  1.        ]])
    The symmetry in the coefficients occurs because all the zeros are on the
    unit circle.
    The coefficients of the denominators of the sections:
    >>> sos[:, 3:]
    array([[ 1.        , -1.32543251,  0.46989499],
           [ 1.        , -1.26117915,  0.6262586 ],
           [ 1.        , -1.25707217,  0.86199667]])
    The next example shows the effect of the `pairing` option.  We have a
    system with three poles and three zeros, so the SOS array will have
    shape (2, 6). The means there is, in effect, an extra pole and an extra
    zero at the origin in the SOS representation.
    >>> z1 = np.array([-1, -0.5-0.5j, -0.5+0.5j])
    >>> p1 = np.array([0.75, 0.8+0.1j, 0.8-0.1j])
    With ``pairing='nearest'`` (the default), we obtain
    >>> signal.zpk2sos(z1, p1, 1)
    array([[ 1.  ,  1.  ,  0.5 ,  1.  , -0.75,  0.  ],
           [ 1.  ,  1.  ,  0.  ,  1.  , -1.6 ,  0.65]])
    The first section has the zeros {-0.5-0.05j, -0.5+0.5j} and the poles
    {0, 0.75}, and the second section has the zeros {-1, 0} and poles
    {0.8+0.1j, 0.8-0.1j}. Note that the extra pole and zero at the origin
    have been assigned to different sections.
    With ``pairing='keep_odd'``, we obtain:
    >>> signal.zpk2sos(z1, p1, 1, pairing='keep_odd')
    array([[ 1.  ,  1.  ,  0.  ,  1.  , -0.75,  0.  ],
           [ 1.  ,  1.  ,  0.5 ,  1.  , -1.6 ,  0.65]])
    The extra pole and zero at the origin are in the same section.
    The first section is, in effect, a first-order section.
    """
    # TODO in the near future:
    # 1. Add SOS capability to `filtfilt`, `freqz`, etc. somehow (#3259).
    # 2. Make `decimate` use `sosfilt` instead of `lfilter`.
    # 3. Make sosfilt automatically simplify sections to first order
    #    when possible. Note this might make `sosfiltfilt` a bit harder (ICs).
    # 4. Further optimizations of the section ordering / pole-zero pairing.
    # See the wiki for other potential issues.

    valid_pairings = ['nearest', 'keep_odd']
    if pairing not in valid_pairings:
        raise ValueError('pairing must be one of %s, not %s'
                         % (valid_pairings, pairing))
    if len(z) == len(p) == 0:
        return array([[k, 0., 0., 1., 0., 0.]])

    # ensure we have the same number of poles and zeros, and make copies
    p = numpy.concatenate((p, numpy.zeros(max(len(z) - len(p), 0))))
    z = numpy.concatenate((z, numpy.zeros(max(len(p) - len(z), 0))))
    n_sections = (max(len(p), len(z)) + 1) // 2
    sos = numpy.zeros((n_sections, 6))

    if len(p) % 2 == 1 and pairing == 'nearest':
        p = numpy.concatenate((p, [0.]))
        z = numpy.concatenate((z, [0.]))
    assert len(p) == len(z)

    # Ensure we have complex conjugate pairs
    # (note that _cplxreal only gives us one element of each complex pair):
    z = numpy.concatenate(_cplxreal(z))
    p = numpy.concatenate(_cplxreal(p))

    p_sos = numpy.zeros((n_sections, 2), numpy.complex128)
    z_sos = numpy.zeros_like(p_sos)
    for si in range(n_sections):
        # Select the next "worst" pole
        p1_idx = numpy.argmin(numpy.abs(1 - numpy.abs(p)))
        p1 = p[p1_idx]
        p = numpy.delete(p, p1_idx)

        # Pair that pole with a zero

        if numpy.isreal(p1) and numpy.isreal(p).sum() == 0:
            # Special case to set a first-order section
            z1_idx = _nearest_real_complex_idx(z, p1, 'real')
            z1 = z[z1_idx]
            z = numpy.delete(z, z1_idx)
            p2 = z2 = 0
        else:
            if not numpy.isreal(p1) and numpy.isreal(z).sum() == 1:
                # Special case to ensure we choose a complex zero to pair
                # with so later (setting up a first-order section)
                z1_idx = _nearest_real_complex_idx(z, p1, 'complex')
                assert not numpy.isreal(z[z1_idx])
            else:
                # Pair the pole with the closest zero (real or complex)
                z1_idx = numpy.argmin(numpy.abs(p1 - z))
            z1 = z[z1_idx]
            z = numpy.delete(z, z1_idx)

            # Now that we have p1 and z1, figure out what p2 and z2 need to be
            if not numpy.isreal(p1):
                if not numpy.isreal(z1):  # complex pole, complex zero
                    p2 = p1.conj()
                    z2 = z1.conj()
                else:  # complex pole, real zero
                    p2 = p1.conj()
                    z2_idx = _nearest_real_complex_idx(z, p1, 'real')
                    z2 = z[z2_idx]
                    assert numpy.isreal(z2)
                    z = numpy.delete(z, z2_idx)
            else:
                if not numpy.isreal(z1):  # real pole, complex zero
                    z2 = z1.conj()
                    p2_idx = _nearest_real_complex_idx(p, z1, 'real')
                    p2 = p[p2_idx]
                    assert numpy.isreal(p2)
                else:  # real pole, real zero
                    # pick the next "worst" pole to use
                    idx = numpy.nonzero(numpy.isreal(p))[0]
                    assert len(idx) > 0
                    p2_idx = idx[numpy.argmin(numpy.abs(numpy.abs(p[idx]) - 1))]
                    p2 = p[p2_idx]
                    # find a real zero to match the added pole
                    assert numpy.isreal(p2)
                    z2_idx = _nearest_real_complex_idx(z, p2, 'real')
                    z2 = z[z2_idx]
                    assert numpy.isreal(z2)
                    z = numpy.delete(z, z2_idx)
                p = numpy.delete(p, p2_idx)
        p_sos[si] = [p1, p2]
        z_sos[si] = [z1, z2]
    assert len(p) == len(z) == 0  # we've consumed all poles and zeros
    del p, z

    # Construct the system, reversing order so the "worst" are last
    p_sos = numpy.reshape(p_sos[::-1], (n_sections, 2))
    z_sos = numpy.reshape(z_sos[::-1], (n_sections, 2))
    gains = numpy.ones(n_sections, numpy.array(k).dtype)
    gains[0] = k
    for si in range(n_sections):
        x = zpk2tf(z_sos[si], p_sos[si], gains[si])
        sos[si] = numpy.concatenate(x)
    return sos

def zpk2tf(z, p, k):
    """
    Return polynomial transfer function representation from zeros and poles
    Parameters
    ----------
    z : array_like
        Zeros of the transfer function.
    p : array_like
        Poles of the transfer function.
    k : float
        System gain.
    Returns
    -------
    b : ndarray
        Numerator polynomial coefficients.
    a : ndarray
        Denominator polynomial coefficients.
    """
    z = numpy.atleast_1d(z)
    k = numpy.atleast_1d(k)
    if len(z.shape) > 1:
        temp = poly(z[0])
        b = zeros((z.shape[0], z.shape[1] + 1), temp.dtype.char)
        if len(k) == 1:
            k = [k[0]] * z.shape[0]
        for i in range(z.shape[0]):
            b[i] = k[i] * poly(z[i])
    else:
        b = k * numpy.poly(z)
    a = numpy.atleast_1d(numpy.poly(p))

    # Use real output if possible. Copied from numpy.poly, since
    # we can't depend on a specific version of numpy.
    if issubclass(b.dtype.type, numpy.complexfloating):
        # if complex roots are all complex conjugates, the roots are real.
        roots = numpy.asarray(z, complex)
        pos_roots = numpy.compress(roots.imag > 0, roots)
        neg_roots = numpy.conjugate(numpy.compress(roots.imag < 0, roots))
        if len(pos_roots) == len(neg_roots):
            if numpy.all(numpy.sort_complex(neg_roots) ==
                         numpy.sort_complex(pos_roots)):
                b = b.real.copy()

    if issubclass(a.dtype.type, numpy.complexfloating):
        # if complex roots are all complex conjugates, the roots are real.
        roots = numpy.asarray(p, complex)
        pos_roots = numpy.compress(roots.imag > 0, roots)
        neg_roots = numpy.conjugate(numpy.compress(roots.imag < 0, roots))
        if len(pos_roots) == len(neg_roots):
            if numpy.all(numpy.sort_complex(neg_roots) ==
                         numpy.sort_complex(pos_roots)):
                a = a.real.copy()

    return b, a

def lp2hp_zpk(z, p, k, wo=1.0):
    r"""
    Transform a lowpass filter prototype to a highpass filter.
    Return an analog high-pass filter with cutoff frequency `wo`
    from an analog low-pass filter prototype with unity cutoff frequency,
    using zeros, poles, and gain ('zpk') representation.
    Parameters
    ----------
    z : array_like
        Zeros of the analog filter transfer function.
    p : array_like
        Poles of the analog filter transfer function.
    k : float
        System gain of the analog filter transfer function.
    wo : float
        Desired cutoff, as angular frequency (e.g., rad/s).
        Defaults to no change.
    Returns
    -------
    z : ndarray
        Zeros of the transformed high-pass filter transfer function.
    p : ndarray
        Poles of the transformed high-pass filter transfer function.
    k : float
        System gain of the transformed high-pass filter.
    See Also
    --------
    lp2lp_zpk, lp2bp_zpk, lp2bs_zpk, bilinear
    lp2hp
    Notes
    -----
    This is derived from the s-plane substitution
    .. math:: s \rightarrow \frac{\omega_0}{s}
    This maintains symmetry of the lowpass and highpass responses on a
    logarithmic scale.
    .. versionadded:: 1.1.0
    """
    z = numpy.atleast_1d(z)
    p = numpy.atleast_1d(p)
    wo = float(wo)

    degree = _relative_degree(z, p)

    # Invert positions radially about unit circle to convert LPF to HPF
    # Scale all points radially from origin to shift cutoff frequency
    z_hp = wo / z
    p_hp = wo / p

    # If lowpass had zeros at infinity, inverting moves them to origin.
    z_hp = numpy.append(z_hp, numpy.zeros(degree))

    # Cancel out gain change caused by inversion
    k_hp = k * numpy.real(numpy.prod(-z) / numpy.prod(-p))

    return z_hp, p_hp, k_hp


def iirfilter(N, Wn, rp=None, rs=None, btype='band', analog=False,
              ftype='butter', output='ba', fs=None):
    """
    IIR digital and analog filter design given order and critical points.
    Design an Nth-order digital or analog filter and return the filter
    coefficients.
    Parameters
    ----------
    N : int
        The order of the filter.
    Wn : array_like
        A scalar or length-2 sequence giving the critical frequencies.
        For digital filters, `Wn` are in the same units as `fs`. By default,
        `fs` is 2 half-cycles/sample, so these are normalized from 0 to 1,
        where 1 is the Nyquist frequency. (`Wn` is thus in
        half-cycles / sample.)
        For analog filters, `Wn` is an angular frequency (e.g., rad/s).
    rp : float, optional
        For Chebyshev and elliptic filters, provides the maximum ripple
        in the passband. (dB)
    rs : float, optional
        For Chebyshev and elliptic filters, provides the minimum attenuation
        in the stop band. (dB)
    btype : {'bandpass', 'lowpass', 'highpass', 'bandstop'}, optional
        The type of filter.  Default is 'bandpass'.
    analog : bool, optional
        When True, return an analog filter, otherwise a digital filter is
        returned.
    ftype : str, optional
        The type of IIR filter to design:
            - Butterworth   : 'butter'
            - Chebyshev I   : 'cheby1'
            - Chebyshev II  : 'cheby2'
            - Cauer/elliptic: 'ellip'
            - Bessel/Thomson: 'bessel'
    output : {'ba', 'zpk', 'sos'}, optional
        Type of output:  numerator/denominator ('ba'), pole-zero ('zpk'), or
        second-order sections ('sos'). Default is 'ba' for backwards
        compatibility, but 'sos' should be used for general-purpose filtering.
    fs : float, optional
        The sampling frequency of the digital system.
        .. versionadded:: 1.2.0
    Returns
    -------
    b, a : ndarray, ndarray
        Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.
        Only returned if ``output='ba'``.
    z, p, k : ndarray, ndarray, float
        Zeros, poles, and system gain of the IIR filter transfer
        function.  Only returned if ``output='zpk'``.
    sos : ndarray
        Second-order sections representation of the IIR filter.
        Only returned if ``output=='sos'``.
    See Also
    --------
    butter : Filter design using order and critical points
    cheby1, cheby2, ellip, bessel
    buttord : Find order and critical points from passband and stopband spec
    cheb1ord, cheb2ord, ellipord
    iirdesign : General filter design using passband and stopband spec
    Notes
    -----
    The ``'sos'`` output parameter was added in 0.16.0.
    Examples
    --------
    Generate a 17th-order Chebyshev II analog bandpass filter from 50 Hz to
    200 Hz and plot the frequency response:
    >>> from scipy import signal
    >>> import matplotlib.pyplot as plt
    >>> b, a = signal.iirfilter(17, [2*np.pi*50, 2*np.pi*200], rs=60,
    ...                         btype='band', analog=True, ftype='cheby2')
    >>> w, h = signal.freqs(b, a, 1000)
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(1, 1, 1)
    >>> ax.semilogx(w / (2*np.pi), 20 * np.log10(np.maximum(abs(h), 1e-5)))
    >>> ax.set_title('Chebyshev Type II bandpass frequency response')
    >>> ax.set_xlabel('Frequency [Hz]')
    >>> ax.set_ylabel('Amplitude [dB]')
    >>> ax.axis((10, 1000, -100, 10))
    >>> ax.grid(which='both', axis='both')
    >>> plt.show()
    Create a digital filter with the same properties, in a system with
    sampling rate of 2000 Hz, and plot the frequency response. (Second-order
    sections implementation is required to ensure stability of a filter of
    this order):
    >>> sos = signal.iirfilter(17, [50, 200], rs=60, btype='band',
    ...                        analog=False, ftype='cheby2', fs=2000,
    ...                        output='sos')
    >>> w, h = signal.sosfreqz(sos, 2000, fs=2000)
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(1, 1, 1)
    >>> ax.semilogx(w, 20 * np.log10(np.maximum(abs(h), 1e-5)))
    >>> ax.set_title('Chebyshev Type II bandpass frequency response')
    >>> ax.set_xlabel('Frequency [Hz]')
    >>> ax.set_ylabel('Amplitude [dB]')
    >>> ax.axis((10, 1000, -100, 10))
    >>> ax.grid(which='both', axis='both')
    >>> plt.show()
    """
    band_dict = {'band': 'bandpass',
                 'bandpass': 'bandpass',
                 'pass': 'bandpass',
                 'bp': 'bandpass',

                 'bs': 'bandstop',
                 'bandstop': 'bandstop',
                 'bands': 'bandstop',
                 'stop': 'bandstop',

                 'l': 'lowpass',
                 'low': 'lowpass',
                 'lowpass': 'lowpass',
                 'lp': 'lowpass',

                 'high': 'highpass',
                 'highpass': 'highpass',
                 'h': 'highpass',
                 'hp': 'highpass',
                 }

    filter_dict = {'butter': [buttap, buttord],
                   'butterworth': [buttap, buttord],
                   }
    ftype, btype, output = [x.lower() for x in (ftype, btype, output)]
    Wn = numpy.asarray(Wn)
    if fs is not None:
        if analog:
            raise ValueError("fs cannot be specified for an analog filter")
        Wn = 2*Wn/fs

    try:
        btype = band_dict[btype]
    except KeyError:
        raise ValueError("'%s' is an invalid bandtype for filter." % btype)

    try:
        typefunc = filter_dict[ftype][0]
    except KeyError:
        raise ValueError("'%s' is not a valid basic IIR filter." % ftype)

    if output not in ['ba', 'zpk', 'sos']:
        raise ValueError("'%s' is not a valid output form." % output)

    if rp is not None and rp < 0:
        raise ValueError("passband ripple (rp) must be positive")

    if rs is not None and rs < 0:
        raise ValueError("stopband attenuation (rs) must be positive")

    # Get analog lowpass prototype
    if typefunc == buttap:
        z, p, k = typefunc(N)
    elif typefunc == besselap:
        z, p, k = typefunc(N, norm=bessel_norms[ftype])
    elif typefunc == cheb1ap:
        if rp is None:
            raise ValueError("passband ripple (rp) must be provided to "
                             "design a Chebyshev I filter.")
        z, p, k = typefunc(N, rp)
    elif typefunc == cheb2ap:
        if rs is None:
            raise ValueError("stopband attenuation (rs) must be provided to "
                             "design an Chebyshev II filter.")
        z, p, k = typefunc(N, rs)
    elif typefunc == ellipap:
        if rs is None or rp is None:
            raise ValueError("Both rp and rs must be provided to design an "
                             "elliptic filter.")
        z, p, k = typefunc(N, rp, rs)
    else:
        raise NotImplementedError("'%s' not implemented in iirfilter." % ftype)

    # Pre-warp frequencies for digital filter design
    if not analog:
        if numpy.any(Wn <= 0) or numpy.any(Wn >= 1):
            if fs is not None:
                raise ValueError("Digital filter critical frequencies "
                                 "must be 0 < Wn < fs/2 (fs={} -> fs/2={})".format(fs, fs/2))
            raise ValueError("Digital filter critical frequencies "
                             "must be 0 < Wn < 1")
        fs = 2.0
        warped = 2 * fs * numpy.tan(numpy.pi * Wn / fs)
    else:
        warped = Wn

    # transform to lowpass, bandpass, highpass, or bandstop
    if btype in ('lowpass', 'highpass'):
        if numpy.size(Wn) != 1:
            raise ValueError('Must specify a single critical frequency Wn for lowpass or highpass filter')

        if btype == 'lowpass':
            z, p, k = lp2lp_zpk(z, p, k, wo=warped)
        elif btype == 'highpass':
            z, p, k = lp2hp_zpk(z, p, k, wo=warped)
    elif btype in ('bandpass', 'bandstop'):
        try:
            bw = warped[1] - warped[0]
            wo = numpy.sqrt(warped[0] * warped[1])
        except IndexError:
            raise ValueError('Wn must specify start and stop frequencies for bandpass or bandstop filter')

        if btype == 'bandpass':
            z, p, k = lp2bp_zpk(z, p, k, wo=wo, bw=bw)
        elif btype == 'bandstop':
            z, p, k = lp2bs_zpk(z, p, k, wo=wo, bw=bw)
    else:
        raise NotImplementedError("'%s' not implemented in iirfilter." % btype)

    # Find discrete equivalent if necessary
    if not analog:
        z, p, k = bilinear_zpk(z, p, k, fs=fs)

    # Transform to proper out type (pole-zero, state-space, numer-denom)
    if output == 'zpk':
        return z, p, k
    elif output == 'ba':
        return zpk2tf(z, p, k)
    elif output == 'sos':
        return zpk2sos(z, p, k)


def sosfreqz(sos, worN=512, whole=False, fs=2*numpy.pi):
    r"""
    Compute the frequency response of a digital filter in SOS format.
    Given `sos`, an array with shape (n, 6) of second order sections of
    a digital filter, compute the frequency response of the system function::
               B0(z)   B1(z)         B{n-1}(z)
        H(z) = ----- * ----- * ... * ---------
               A0(z)   A1(z)         A{n-1}(z)
    for z = exp(omega*1j), where B{k}(z) and A{k}(z) are numerator and
    denominator of the transfer function of the k-th second order section.
    Parameters
    ----------
    sos : array_like
        Array of second-order filter coefficients, must have shape
        ``(n_sections, 6)``. Each row corresponds to a second-order
        section, with the first three columns providing the numerator
        coefficients and the last three providing the denominator
        coefficients.
    worN : {None, int, array_like}, optional
        If a single integer, then compute at that many frequencies (default is
        N=512).  Using a number that is fast for FFT computations can result
        in faster computations (see Notes of `freqz`).
        If an array_like, compute the response at the frequencies given (must
        be 1-D). These are in the same units as `fs`.
    whole : bool, optional
        Normally, frequencies are computed from 0 to the Nyquist frequency,
        fs/2 (upper-half of unit-circle). If `whole` is True, compute
        frequencies from 0 to fs.
    fs : float, optional
        The sampling frequency of the digital system. Defaults to 2*pi
        radians/sample (so w is from 0 to pi).
        .. versionadded:: 1.2.0
    Returns
    -------
    w : ndarray
        The frequencies at which `h` was computed, in the same units as `fs`.
        By default, `w` is normalized to the range [0, pi) (radians/sample).
    h : ndarray
        The frequency response, as complex numbers.
    See Also
    --------
    freqz, sosfilt
    Notes
    -----
    .. versionadded:: 0.19.0
    Examples
    --------
    Design a 15th-order bandpass filter in SOS format.
    >>> from scipy import signal
    >>> sos = signal.ellip(15, 0.5, 60, (0.2, 0.4), btype='bandpass',
    ...                    output='sos')
    Compute the frequency response at 1500 points from DC to Nyquist.
    >>> w, h = signal.sosfreqz(sos, worN=1500)
    Plot the response.
    >>> import matplotlib.pyplot as plt
    >>> plt.subplot(2, 1, 1)
    >>> db = 20*np.log10(np.maximum(np.abs(h), 1e-5))
    >>> plt.plot(w/np.pi, db)
    >>> plt.ylim(-75, 5)
    >>> plt.grid(True)
    >>> plt.yticks([0, -20, -40, -60])
    >>> plt.ylabel('Gain [dB]')
    >>> plt.title('Frequency Response')
    >>> plt.subplot(2, 1, 2)
    >>> plt.plot(w/np.pi, np.angle(h))
    >>> plt.grid(True)
    >>> plt.yticks([-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi],
    ...            [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    >>> plt.ylabel('Phase [rad]')
    >>> plt.xlabel('Normalized frequency (1.0 = Nyquist)')
    >>> plt.show()
    If the same filter is implemented as a single transfer function,
    numerical error corrupts the frequency response:
    >>> b, a = signal.ellip(15, 0.5, 60, (0.2, 0.4), btype='bandpass',
    ...                    output='ba')
    >>> w, h = signal.freqz(b, a, worN=1500)
    >>> plt.subplot(2, 1, 1)
    >>> db = 20*np.log10(np.maximum(np.abs(h), 1e-5))
    >>> plt.plot(w/np.pi, db)
    >>> plt.ylim(-75, 5)
    >>> plt.grid(True)
    >>> plt.yticks([0, -20, -40, -60])
    >>> plt.ylabel('Gain [dB]')
    >>> plt.title('Frequency Response')
    >>> plt.subplot(2, 1, 2)
    >>> plt.plot(w/np.pi, np.angle(h))
    >>> plt.grid(True)
    >>> plt.yticks([-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi],
    ...            [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    >>> plt.ylabel('Phase [rad]')
    >>> plt.xlabel('Normalized frequency (1.0 = Nyquist)')
    >>> plt.show()
    """

    sos, n_sections = _validate_sos(sos)
    if n_sections == 0:
        raise ValueError('Cannot compute frequencies with no sections')
    h = 1.
    for row in sos:
        w, rowh = freqz(row[:3], row[3:], worN=worN, whole=whole, fs=fs)
        h *= rowh
    return w, h

def _is_int_type(x):
    """
    Check if input is of a scalar integer type (so ``5`` and ``array(5)`` will
    pass, while ``5.0`` and ``array([5])`` will fail.
    """
    if numpy.ndim(x) != 0:
        # Older versions of NumPy did not raise for np.array([1]).__index__()
        # This is safe to remove when support for those versions is dropped
        return False
    try:
        operator.index(x)
    except TypeError:
        return False
    else:
        return True

def freqz(b, a=1, worN=512, whole=False, plot=None, fs=2*numpy.pi, include_nyquist=False):
    """
    Compute the frequency response of a digital filter.
    Given the M-order numerator `b` and N-order denominator `a` of a digital
    filter, compute its frequency response::
                 jw                 -jw              -jwM
        jw    B(e  )    b[0] + b[1]e    + ... + b[M]e
     H(e  ) = ------ = -----------------------------------
                 jw                 -jw              -jwN
              A(e  )    a[0] + a[1]e    + ... + a[N]e
    Parameters
    ----------
    b : array_like
        Numerator of a linear filter. If `b` has dimension greater than 1,
        it is assumed that the coefficients are stored in the first dimension,
        and ``b.shape[1:]``, ``a.shape[1:]``, and the shape of the frequencies
        array must be compatible for broadcasting.
    a : array_like
        Denominator of a linear filter. If `b` has dimension greater than 1,
        it is assumed that the coefficients are stored in the first dimension,
        and ``b.shape[1:]``, ``a.shape[1:]``, and the shape of the frequencies
        array must be compatible for broadcasting.
    worN : {None, int, array_like}, optional
        If a single integer, then compute at that many frequencies (default is
        N=512). This is a convenient alternative to::
            np.linspace(0, fs if whole else fs/2, N, endpoint=include_nyquist)
        Using a number that is fast for FFT computations can result in
        faster computations (see Notes).
        If an array_like, compute the response at the frequencies given.
        These are in the same units as `fs`.
    whole : bool, optional
        Normally, frequencies are computed from 0 to the Nyquist frequency,
        fs/2 (upper-half of unit-circle). If `whole` is True, compute
        frequencies from 0 to fs. Ignored if w is array_like.
    plot : callable
        A callable that takes two arguments. If given, the return parameters
        `w` and `h` are passed to plot. Useful for plotting the frequency
        response inside `freqz`.
    fs : float, optional
        The sampling frequency of the digital system. Defaults to 2*pi
        radians/sample (so w is from 0 to pi).
        .. versionadded:: 1.2.0
    include_nyquist : bool, optional
        If `whole` is False and `worN` is an integer, setting `include_nyquist` to True
        will include the last frequency (Nyquist frequency) and is otherwise ignored.
        .. versionadded:: 1.5.0
    Returns
    -------
    w : ndarray
        The frequencies at which `h` was computed, in the same units as `fs`.
        By default, `w` is normalized to the range [0, pi) (radians/sample).
    h : ndarray
        The frequency response, as complex numbers.
    See Also
    --------
    freqz_zpk
    sosfreqz
    Notes
    -----
    Using Matplotlib's :func:`matplotlib.pyplot.plot` function as the callable
    for `plot` produces unexpected results, as this plots the real part of the
    complex transfer function, not the magnitude.
    Try ``lambda w, h: plot(w, np.abs(h))``.
    A direct computation via (R)FFT is used to compute the frequency response
    when the following conditions are met:
    1. An integer value is given for `worN`.
    2. `worN` is fast to compute via FFT (i.e.,
       `next_fast_len(worN) <scipy.fft.next_fast_len>` equals `worN`).
    3. The denominator coefficients are a single value (``a.shape[0] == 1``).
    4. `worN` is at least as long as the numerator coefficients
       (``worN >= b.shape[0]``).
    5. If ``b.ndim > 1``, then ``b.shape[-1] == 1``.
    For long FIR filters, the FFT approach can have lower error and be much
    faster than the equivalent direct polynomial calculation.
    Examples
    --------
    >>> from scipy import signal
    >>> b = signal.firwin(80, 0.5, window=('kaiser', 8))
    >>> w, h = signal.freqz(b)
    >>> import matplotlib.pyplot as plt
    >>> fig, ax1 = plt.subplots()
    >>> ax1.set_title('Digital filter frequency response')
    >>> ax1.plot(w, 20 * np.log10(abs(h)), 'b')
    >>> ax1.set_ylabel('Amplitude [dB]', color='b')
    >>> ax1.set_xlabel('Frequency [rad/sample]')
    >>> ax2 = ax1.twinx()
    >>> angles = np.unwrap(np.angle(h))
    >>> ax2.plot(w, angles, 'g')
    >>> ax2.set_ylabel('Angle (radians)', color='g')
    >>> ax2.grid()
    >>> ax2.axis('tight')
    >>> plt.show()
    Broadcasting Examples
    Suppose we have two FIR filters whose coefficients are stored in the
    rows of an array with shape (2, 25). For this demonstration, we'll
    use random data:
    >>> np.random.seed(42)
    >>> b = np.random.rand(2, 25)
    To compute the frequency response for these two filters with one call
    to `freqz`, we must pass in ``b.T``, because `freqz` expects the first
    axis to hold the coefficients. We must then extend the shape with a
    trivial dimension of length 1 to allow broadcasting with the array
    of frequencies.  That is, we pass in ``b.T[..., np.newaxis]``, which has
    shape (25, 2, 1):
    >>> w, h = signal.freqz(b.T[..., np.newaxis], worN=1024)
    >>> w.shape
    (1024,)
    >>> h.shape
    (2, 1024)
    Now, suppose we have two transfer functions, with the same numerator
    coefficients ``b = [0.5, 0.5]``. The coefficients for the two denominators
    are stored in the first dimension of the 2-D array  `a`::
        a = [   1      1  ]
            [ -0.25, -0.5 ]
    >>> b = np.array([0.5, 0.5])
    >>> a = np.array([[1, 1], [-0.25, -0.5]])
    Only `a` is more than 1-D. To make it compatible for
    broadcasting with the frequencies, we extend it with a trivial dimension
    in the call to `freqz`:
    >>> w, h = signal.freqz(b, a[..., np.newaxis], worN=1024)
    >>> w.shape
    (1024,)
    >>> h.shape
    (2, 1024)
    """
    b = numpy.atleast_1d(b)
    a = numpy.atleast_1d(a)

    if worN is None:
        # For backwards compatibility
        worN = 512

    h = None

    if _is_int_type(worN):
        N = operator.index(worN)
        del worN
        if N < 0:
            raise ValueError('worN must be nonnegative, got %s' % (N,))
        lastpoint = 2 * numpy.pi if whole else numpy.pi
        # if include_nyquist is true and whole is false, w should include end point
        w = numpy.linspace(0, lastpoint, N, endpoint=include_nyquist and not whole)
        if (a.size == 1 and N >= b.shape[0] and
                sp_fft.next_fast_len(N) == N and
                (b.ndim == 1 or (b.shape[-1] == 1))):
            # if N is fast, 2 * N will be fast, too, so no need to check
            n_fft = N if whole else N * 2
            if numpy.isrealobj(b) and numpy.isrealobj(a):
                fft_func = sp_fft.rfft
            else:
                fft_func = sp_fft.fft
            h = fft_func(b, n=n_fft, axis=0)[:N]
            h /= a
            if fft_func is sp_fft.rfft and whole:
                # exclude DC and maybe Nyquist (no need to use axis_reverse
                # here because we can build reversal with the truncation)
                stop = -1 if n_fft % 2 == 1 else -2
                h_flip = slice(stop, 0, -1)
                h = numpy.concatenate((h, h[h_flip].conj()))
            if b.ndim > 1:
                # Last axis of h has length 1, so drop it.
                h = h[..., 0]
                # Rotate the first axis of h to the end.
                h = numpy.rollaxis(h, 0, h.ndim)
    else:
        w = atleast_1d(worN)
        del worN
        w = 2*numpy.pi*w/fs

    if h is None:  # still need to compute using freqs w
        zm1 = numpy.exp(-1j * w)
        h = (npp_polyval(zm1, b, tensor=False) /
             npp_polyval(zm1, a, tensor=False))

    w = w*fs/(2*numpy.pi)

    if plot is not None:
        plot(w, h)

    return w, h

def _validate_sos(sos):
    """Helper to validate a SOS input"""
    sos = numpy.atleast_2d(sos)
    if sos.ndim != 2:
        raise ValueError('sos array must be 2D')
    n_sections, m = sos.shape
    if m != 6:
        raise ValueError('sos array must be shape (n_sections, 6)')
    if not (sos[:, 3] == 1).all():
        raise ValueError('sos[:, 3] should be all ones')
    return sos, n_sections

#y = CreateWhitenoise(44100,512)
#y3 = CreateSquarewave(44100,1000,512)
sine_full = CreateSinewave(44100,1000,512)
music = MonoWavToNumpy16BitInt('testmusic_mono.wav')
music_copy = copy.deepcopy(music)
#sine,sine2,sine3,sine4 = numpy.split(sine_full,4)


#Numpy16BitIntToMonoWav('test.wav',y22)
start = timeit.default_timer()
sos = iirfilter(2, [200], rs=60, btype='high', analog = False, ftype = 'butter', fs = 44100, output = 'sos')
w, h = sosfreqz(sos, 44100, fs=44100)
stop = timeit.default_timer()
print('Time: ', (stop - start)*1000, 'ms')

fig = pyplot.figure()
ax = fig.add_subplot(1, 1, 1)
ax.semilogx(w, 20 * numpy.log10(numpy.maximum(abs(h), 1e-5)))
ax.set_title('Butterworth highpass frequency response')
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel('Amplitude [dB]')
ax.axis((10, 20000, -100, 10))
ax.grid(which='both', axis='both')
pyplot.show()
#wave = sine_full
#wave,freq=EQ(sine_full)
#limiter.limit(sine2)
#limiter.limit(sine3)
#limiter.limit(sine4)
#sine_add = numpy.append(sine,sine2)
#sine_add = numpy.append(sine_add,sine3)
#sine_add = numpy.append(sine_add,sine4)
#Numpy16BitIntToMonoWav44kHz("output.wav",sine)


#pyplot.plot(XaxisForMatplotlib(sine),sine)
#pyplot.plot(XaxisForMatplotlib(y3),y3)
#pyplot.plot(XaxisForMatplotlib(sine), sine)
pyplot.plot(wave)
#pyplot.plot(XaxisForMatplotlib(y6),y6)


pyplot.show()



sys.exit()



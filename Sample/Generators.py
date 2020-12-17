import numpy
import config

def CreateSinewave(sin_frequency, sin_length_in_samples, sin_sample_rate=config.sampling_rate):
    sin_time_array = numpy.arange(sin_length_in_samples)
    sin_amplitude_array = numpy.float32(numpy.sin(2 * numpy.pi * sin_frequency*sin_time_array/sin_sample_rate))
    return (sin_amplitude_array)

def CreateSquarewave(square_frequency, square_length_in_samples,square_sample_rate=config.sampling_rate):
    square_time_array = numpy.arange(square_length_in_samples)
    square_amplitude_array = numpy.float32(numpy.sin(2 * numpy.pi * square_frequency * square_time_array / square_sample_rate))
    square_amplitude_array = numpy.where(square_amplitude_array>0,1.0,-1.0)
    return (square_amplitude_array)

def CreateWhitenoise(noise_length_in_samples,sample_rate=config.sampling_rate):
    whitenoise_time_array = numpy.arange(noise_length_in_samples)
    freqs = numpy.abs(numpy.fft.fftfreq(noise_length_in_samples, 1/sample_rate))
    f = numpy.zeros(noise_length_in_samples)
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

    whitenoise_amplitude_array = numpy.float32(fftnoise(f))

    return whitenoise_amplitude_array



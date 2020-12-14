import numpy

def CreateSinewave(sin_sample_rate, sin_frequency, sin_buffer_size):
    sin_time_array = numpy.arange(sin_buffer_size)
    sin_amplitude_array = numpy.float32(numpy.sin(2 * numpy.pi * sin_frequency*sin_time_array/sin_sample_rate))
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

    whitenoise_amplitude_array = numpy.float32(fftnoise(f))

    return whitenoise_time_array,whitenoise_amplitude_array



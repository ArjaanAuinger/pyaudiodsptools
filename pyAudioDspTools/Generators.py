import numpy
from . import config


def CreateSinewave(sin_frequency, sin_length_in_samples):
    """Generates a sine wave with selected properties.

    Parameters
    ----------
    sin_frequency : int
        The frequency of the sine wave.
    sin_length_in_samples : int
        The lenght of the sine wave in samples. Is your sin_sample_rate is 44100 and your sin_length_in_samples is set to
        44100 your sine wave signal will be exactly 1 second long for example
    sin_sample_rate : int
        Is set to config.sampling_rate from your config.py by default. Use pyAudioDspTools.config.sampling_rate=48000 in your script
        to change your sampling rate globally to 48000 hertz for example.

    Returns
    -------
    numpy array
        The created array

    """
    sin_time_array = numpy.arange(sin_length_in_samples)
    sin_amplitude_array = numpy.float32(numpy.sin(2 * numpy.pi * sin_frequency * sin_time_array / config.sampling_rate))
    return sin_amplitude_array


def CreateSquarewave(square_frequency, square_length_in_samples):
    """Generates a square wave with selected properties.

    Parameters
    ----------
    square_frequency : int
        The frequency of the sine wave.
    square_length_in_samples : int
        The lenght of the square wave in samples. Is your square_sample_rate is 44100 and your square_length_in_samples
        is set to 44100 your square wave signal will be exactly 1 second long for example
    square_sample_rate : int
        Is set to config.sampling_rate from your config.py by default. Use pyAudioDspTools.config.sampling_rate=48000 in your script
        to change your sampling rate globally to 48000 hertz for example.

    Returns
    -------
    numpy array
        The created array

    """
    square_time_array = numpy.arange(square_length_in_samples)
    square_amplitude_array = numpy.float32(
        numpy.sin(2 * numpy.pi * square_frequency * square_time_array / config.sampling_rate))
    square_amplitude_array = numpy.where(square_amplitude_array > 0, 1.0, -1.0)
    return square_amplitude_array


def CreateWhitenoise(noise_length_in_samples):
    """Generates noise with selected properties.

    Parameters
    ----------
    noise_length_in_samples : int
        The lenght of the sine wave in samples. Is your square_sample_rate is 44100 and your square_length_in_samples
        is set to 44100 your noise signal will be exactly 1 second long for example
    square_sample_rate : int
        Is set to config.sampling_rate from your config.py by default. Use pyAudioDspTools.config.sampling_rate=48000 in your script
        to change your sampling rate globally to 48000 hertz for example.

    Returns
    -------
    numpy array
        The created array

    """
    whitenoise_time_array = numpy.arange(noise_length_in_samples)
    freqs = numpy.abs(numpy.fft.fftfreq(noise_length_in_samples, 1 / config.sampling_rate))
    f = numpy.zeros(noise_length_in_samples)
    idx = numpy.where(numpy.logical_and(freqs >= 20, freqs <= 20000))[0]
    f[idx] = 1

    def fftnoise(f):
        f = numpy.array(f, dtype='complex')
        Np = (len(f) - 1) // 2
        phases = numpy.random.rand(Np) * 2 * numpy.pi
        phases = numpy.cos(phases) + 1j * numpy.sin(phases)
        f[1:Np + 1] *= phases
        f[-1:-1 - Np:-1] = numpy.conj(f[1:Np + 1])
        return (numpy.fft.ifft(f).real * 5)

    whitenoise_amplitude_array = numpy.float32(fftnoise(f))

    return whitenoise_amplitude_array

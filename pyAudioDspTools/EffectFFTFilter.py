from .config import chunk_size, sampling_rate
import numpy


class CreateHighCutFilter:
    """Creating a FFT filter audio-effect class/device.

    Cuts the upper frequencies of a signal.
    Is overloaded with basic settings.
    This class introduces latency equal to chunk_size.

    Parameters
    ----------
    cutoff_frequency : int or float
        Sets the rolloff frequency for the high cut filter.

    """
    def __init__(self, cutoff_frequency=8000):
        #self.chunk_size = chunk_size
        self.fS = sampling_rate  # Sampling rate.
        self.fH = cutoff_frequency  # Cutoff frequency.
        self.filter_length = (chunk_size // 2) - 1  # Filter length, must be odd.

        self.array_slice_value_start = chunk_size + (self.filter_length // 2)
        self.array_slice_value_end = chunk_size - (self.filter_length // 2)

        # Compute sinc filter.
        self.sinc_filter = numpy.sinc(
            2 * self.fH / self.fS * (numpy.arange(self.filter_length) - (self.filter_length - 1) / 2))
        # pyplot.plot(self.sinc_filter)

        # Apply window.
        self.sinc_filter *= numpy.blackman(self.filter_length)
        # pyplot.plot(self.sinc_filter)

        # Normalize to get unity gain.
        self.sinc_filter /= numpy.sum(self.sinc_filter)

        self.filtered_signal = numpy.zeros(chunk_size * 3)
        self.float32_array_input_1 = numpy.zeros(chunk_size)
        self.float32_array_input_2 = numpy.zeros(chunk_size)
        self.float32_array_input_3 = numpy.zeros(chunk_size)

        self.cut_size = numpy.int16((self.filter_length - 1) / 2)
        self.sinc_filter = numpy.append(self.sinc_filter, numpy.zeros(chunk_size - self.filter_length + 1))
        self.sinc_filter = numpy.append(self.sinc_filter, numpy.zeros(((len(self.sinc_filter) * 2) - 3)))
        self.sinc_filter = numpy.fft.fft(self.sinc_filter)

    def apply(self, float32_array_input):
        """Applying the filter to a numpy-array

        Parameters
        ----------
        float_array_input : float
            The array, which the effect should be applied on.

        Returns
        -------
        float
            The previously processed array, should be the exact same size as the input array

        """
        self.float32_array_input_3 = self.float32_array_input_2
        self.float32_array_input_2 = self.float32_array_input_1
        self.float32_array_input_1 = float32_array_input

        self.filtered_signal = numpy.concatenate(
            (self.float32_array_input_3, self.float32_array_input_2, self.float32_array_input_1), axis=None)

        self.filtered_signal = numpy.fft.fft(self.filtered_signal)
        self.filtered_signal = self.filtered_signal * self.sinc_filter
        self.filtered_signal = numpy.fft.ifft(self.filtered_signal)
        self.filtered_signal = self.filtered_signal[self.array_slice_value_start:-self.array_slice_value_end]

        return self.filtered_signal.real.astype('float32')


class CreateLowCutFilter:
    """Creating a FFT filter audio-effect class/device.

    Cuts the lower frequencies of a signal.
    Is overloaded with basic settings.
    This class introduces latency equal to chunk_size.

    Parameters
    ----------
    cutoff_frequency : int or float
        Sets the rolloff frequency for the high cut filter.

    """
    def __init__(self, cutoff_frequency=160):
        #self.chunk_size = chunk_size
        self.fS = sampling_rate  # Sampling rate.
        self.fH = cutoff_frequency  # Cutoff frequency.
        self.filter_length = (chunk_size // 2) - 1  # Filter length, must be odd.

        self.array_slice_value_start = chunk_size + (self.filter_length // 2)
        self.array_slice_value_end = chunk_size - (self.filter_length // 2)

        # Compute sinc filter.
        self.sinc_filter = numpy.sinc(
            2 * self.fH / self.fS * (numpy.arange(self.filter_length) - (self.filter_length - 1) / 2))

        # Apply window.
        self.sinc_filter *= numpy.blackman(self.filter_length)

        # Normalize to get unity gain.
        self.sinc_filter /= numpy.sum(self.sinc_filter)
        # print(len(self.sinc_filter))

        # Spectral inversion to create Lowcut from Highcut
        self.sinc_filter = -self.sinc_filter
        self.sinc_filter[(self.filter_length - 1) // 2] += 1

        self.filtered_signal = numpy.zeros(chunk_size * 3)
        self.float32_array_input_1 = numpy.zeros(chunk_size)
        self.float32_array_input_2 = numpy.zeros(chunk_size)
        self.float32_array_input_3 = numpy.zeros(chunk_size)

        self.cut_size = numpy.int16((self.filter_length - 1) / 2)
        self.sinc_filter = numpy.append(self.sinc_filter, numpy.zeros(chunk_size - self.filter_length + 1))
        self.sinc_filter = numpy.append(self.sinc_filter, numpy.zeros(((len(self.sinc_filter) * 2) - 3)))
        self.sinc_filter = numpy.fft.fft(self.sinc_filter)

    def apply(self, float32_array_input):
        """Applying the filter to a numpy-array

        Parameters
        ----------
        float_array_input : float
            The array, which the effect should be applied on.

        Returns
        -------
        float
            The previously processed array, should be the exact same size as the input array

        """
        self.float32_array_input_3 = self.float32_array_input_2
        self.float32_array_input_2 = self.float32_array_input_1
        self.float32_array_input_1 = float32_array_input

        self.filtered_signal = numpy.concatenate(
            (self.float32_array_input_3, self.float32_array_input_2, self.float32_array_input_1), axis=None)

        self.filtered_signal = numpy.fft.fft(self.filtered_signal)
        self.filtered_signal = (self.filtered_signal * self.sinc_filter)
        self.filtered_signal = numpy.fft.ifft(self.filtered_signal)
        self.filtered_signal = self.filtered_signal[self.array_slice_value_start:-self.array_slice_value_end]

        return self.filtered_signal.real.astype('float32')

import numpy
import copy

from .config import chunk_size, sampling_rate

class CreateTremolo:
    """Creating a tremolo audio-effect class/device

    Creates a LFO and applies it to the input array to modulate power.

    Parameters
    ----------
    tremolo_depth : float
        Sets the depth of the effect. Must be a value between >0 and <1.0
    lfo_in_hertz : float
        Sets the cycle of the LFO in seconds.

    """
    def __init__(self,tremolo_depth=0.4,lfo_in_hertz=4.5):
        self.sin_sample_rate = sampling_rate
        self.sin_time_array = numpy.arange(numpy.float32(self.sin_sample_rate/lfo_in_hertz))
        self.sin_lfo = numpy.float32((((numpy.sin(2 * numpy.pi * lfo_in_hertz*self.sin_time_array/self.sin_sample_rate)
                                        /2)+0.5)*tremolo_depth)+(1-tremolo_depth))
        self.lfo_length = len(self.sin_lfo)
        self.sin_lfo_copy = copy.deepcopy(self.sin_lfo)

    def apply(self,float_array_input):
        """Applying the Tremolo to a numpy-array.

        Parameters
        ----------
        float_array_input : float
            The array, which the effect should be applied on.

        Returns
        -------
        float
            The processed array, should be the exact same size as the input array

        """
        current_input_lenght = len(float_array_input)
        while len(self.sin_lfo_copy) < len(float_array_input):
            self.sin_lfo_copy = numpy.append(self.sin_lfo_copy,self.sin_lfo)
        self.sin_lfo_chunk = self.sin_lfo_copy[:current_input_lenght]
        self.sin_lfo_copy = self.sin_lfo_copy[-(len(self.sin_lfo_copy)-current_input_lenght):]
        float_array_output = numpy.multiply(float_array_input, self.sin_lfo_chunk, dtype='float32', casting='unsafe')
        return float_array_output

    def reset(self):
        """Resets the LFO of the Tremolo.

        Parameters
        ----------
        None : None

        """
        self.sin_lfo_copy = copy.deepcopy(self.sin_lfo)
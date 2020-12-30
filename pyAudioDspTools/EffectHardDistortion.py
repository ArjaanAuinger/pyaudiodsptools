import numpy
import copy

class CreateHardDistortion():
    """Creating a distortion audio-effect class/device.

    Its a wave-shaper and messes with dynamic range, but doesn't introduce latency.

    Parameters
    ----------
    None : None

    """
    def __init__(self):
        self.linear_limit = 0.8

    def apply(self,float_array_input):
        """Applying the distortion to a numpy-array.

        Parameters
        ----------
        float_array_input : float
            The array, which the effect should be applied on.

        Returns
        -------
        float
            The processed array, should be the exact same size as the input array

        """
        hard_limit = 1.0
        linear_limit = 0.8
        clip_limit = linear_limit + float(numpy.pi / 2 * (hard_limit - linear_limit))
        sign = copy.deepcopy(float_array_input)
        sign = numpy.where(float_array_input>=0,1,-1)
        amplitude = numpy.absolute(float_array_input)
        amplitude = numpy.where(amplitude <= linear_limit, amplitude,hard_limit*sign)
        scale = hard_limit - linear_limit
        compression = scale * numpy.sin(numpy.float32(amplitude - linear_limit) / scale)
        float_array_output = numpy.float32((linear_limit + numpy.float32(compression)) * sign)
        return float_array_output
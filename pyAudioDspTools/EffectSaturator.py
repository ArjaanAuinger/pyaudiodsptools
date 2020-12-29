import numpy
#import copy

class CreateSaturator:
    """Creating a saturator audio-effect class/device

    Is a wave-shaper and messes with dynamic range, but doesn't introduce latency.

    Parameters
    ----------
    saturation_threshold_in_db : int or float
        Sets the threshold when the saturator becomes active. Must be negative.
    makeup_gain : float
        Makeup for Volume-loss due to wave-shaping in decibel.
    mode : string
        The mode of the Saturator. Can be 'hard' or 'soft'

    """
    def __init__(self, saturation_threshold_in_db=-20.0, makeup_gain=2.0, mode='hard'):
        self.saturation_coeff = 10 ** (saturation_threshold_in_db/20)
        self.makeup_gain = makeup_gain
        if(mode=='soft'):
            self.mode = 2
        if(mode=='hard'):
            self.mode = 1

    def apply(self, float_array_input):
        """Applying the Saturator to a numpy-array

        Parameters
        ----------
        float_array_input : float
            The array, which the effect should be applied on.

        Returns
        -------
        float
            The processed array, should be the exact same size as the input array

        """
        remember_negative = numpy.where(float_array_input<0,True,False)
        float_array_input = numpy.abs(float_array_input)

        #Processing the array
        float_array_input = numpy.where(float_array_input>self.saturation_coeff, self.saturation_coeff + (float_array_input-self.saturation_coeff)/(1+((float_array_input-self.saturation_coeff)/(1-self.saturation_coeff))**self.mode), float_array_input)
        float_array_input = numpy.where(float_array_input>1.0, (self.saturation_coeff+1)/2,float_array_input)
        float_array_input = numpy.where(remember_negative,-float_array_input,float_array_input)
        float_array_output = (10**(self.makeup_gain/20)*float_array_input)
        return float_array_output

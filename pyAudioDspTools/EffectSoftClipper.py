import numpy

class CreateSoftClipper:
    """Creating a limiter-kind audio-effect class/device

    Is a wave-shaper and messes with dynamic range, but doesn't introduce latency.

    Parameters
    ----------
     drive : float
            A value between 0.0 and 1.0, 0.0 meaning no wave shaping at all and 1.0 full drive.

    Notes
    -----
    * You can go beyond 1.0, but I designed it to be at the sweet spot. Go to 70.0 if you want, but be warned.

    """
    def __init__(self,drive = 0.44):
        self.placeholder = True
        self.drive = drive + 1


    def apply(self, float_array_input):
        """Applying the Soft Clipper to a numpy-array

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

        float_array_input = numpy.clip(float_array_input,-1.0,1.0)
        #float_array_input = -1*(numpy.sqrt(float_array_input)*float_array_input-1)**2 + 1
        float_array_input = -1*(numpy.abs(float_array_input-1))**self.drive + 1

        float_array_input = numpy.where(remember_negative, -float_array_input, float_array_input)
        return float_array_input
import numpy
import copy

class CreateTremolo:
    def __init__(self,tremolo_depth,lfo_in_hertz=4.5):
        self.sin_sample_rate = 44100
        self.sin_time_array = numpy.arange(numpy.float32(self.sin_sample_rate/lfo_in_hertz))
        self.sin_lfo = numpy.float32((((numpy.sin(2 * numpy.pi * lfo_in_hertz*self.sin_time_array/self.sin_sample_rate)
                                        /2)+0.5)*tremolo_depth)+(1-tremolo_depth))
        self.lfo_length = len(self.sin_lfo)
        self.sin_lfo_copy = copy.deepcopy(self.sin_lfo)

    def apply(self,float_array_input):
        current_input_lenght = len(float_array_input)
        while len(self.sin_lfo_copy) < len(float_array_input):
            self.sin_lfo_copy = numpy.append(self.sin_lfo_copy,self.sin_lfo)
        self.sin_lfo_chunk = self.sin_lfo_copy[:current_input_lenght]
        self.sin_lfo_copy = self.sin_lfo_copy[-(len(self.sin_lfo_copy)-current_input_lenght):]
        float_array_output = numpy.multiply(float_array_input,self.sin_lfo_chunk, out=int_array_input, dtype='float32', casting='unsafe')
        return float_array_output

    def tremoloreset(self):
        self.sin_lfo_copy = copy.deepcopy(self.sin_lfo)
import numpy
import math

class CreateDelay:
    def __init__(self,time_in_samples=22050,feedback_loops=2):
        #self.max_samples = (((44100/1000)*time_in_ms)*feedback_loops)
        self.time_in_samples = time_in_samples
        self.max_samples = time_in_samples*(feedback_loops+2)
        self.delay_buffer = numpy.zeros(int(self.max_samples), dtype="float32")
        self.feedback_ramp = numpy.linspace(0.5,0.1,num=feedback_loops,dtype="float32")

    def applydelay(self,float32_array_input):

        for counter in range(len(self.feedback_ramp)):
            processed_input = float32_array_input * self.feedback_ramp[counter]
            start_index = self.time_in_samples*(counter+1)
            end_index = (self.time_in_samples*(counter+1))+len(float32_array_input)
            self.delay_buffer[start_index:end_index] += processed_input

        float32_array_input += self.delay_buffer[0:len(float32_array_input)]
        self.delay_buffer = self.delay_buffer[len(float32_array_input):len(self.delay_buffer)]
        self.delay_buffer = numpy.append(self.delay_buffer,numpy.zeros(len(float32_array_input),dtype="float32"))

        return(float32_array_input)
"""
    chunks = 1
    if len(float32_array_input) > self.time_in_samples:
        chunks = math.ceil(len(float32_array_input) / self.time_in_samples)
        float32_array_input = numpy.split(float32_array_input, chunks)
    for float32_array_input in float32_array_input:
"""
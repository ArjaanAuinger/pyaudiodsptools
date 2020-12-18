import numpy
import math
import config
import EffectFFTFilter

class CreateDelay:
    def __init__(self,time_in_ms=500,feedback_loops=2,lowcut_filter_frequency=40,highcut_filter_frequency=12000,use_lowcut_filter=False,use_highcut_filter=False,wet=False):
        self.time_in_samples = numpy.int(time_in_ms*(config.sampling_rate/1000))
        self.wet = wet
        self.max_samples = self.time_in_samples*(feedback_loops+2)
        self.delay_buffer = numpy.zeros(int(self.max_samples), dtype="float32")
        self.feedback_ramp = numpy.linspace(0.5,0.1,num=feedback_loops,dtype="float32")
        self.use_lowcut_filter = use_lowcut_filter
        self.use_highcut_filter = use_highcut_filter
        self.LowCutFilter = EffectFFTFilter.CreateLowCutFilter(lowcut_filter_frequency)
        self.HighcutFilter = EffectFFTFilter.CreateHighCutFilter(highcut_filter_frequency)

    def apply(self,float32_array_input):
        if (self.use_lowcut_filter == True):
            float32_array_input = self.LowCutFilter.applylowcutfilter(float32_array_input)
        if (self.use_highcut_filter == True):
            float32_array_input = self.HighcutFilter.applyhighcutfilter(float32_array_input)

        for counter in range(len(self.feedback_ramp)):
            processed_input = float32_array_input * self.feedback_ramp[counter]
            start_index = self.time_in_samples*(counter+1)
            end_index = (self.time_in_samples*(counter+1))+len(float32_array_input)
            self.delay_buffer[start_index:end_index] += processed_input

        if (self.wet == False):
            float32_array_input += self.delay_buffer[0:len(float32_array_input)]
        else:
            float32_array_input = self.delay_buffer[0:len(float32_array_input)]

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
import numpy
import EffectFilter


class CreateReverb:
    def __init__(self,time_in_samples=6000,early_feedback_loops=20,late_feedback_loops=20):
        self.time_in_samples = time_in_samples
        self.max_samples = time_in_samples*(early_feedback_loops+2)
        self.delay_buffer = numpy.zeros(int(self.max_samples), dtype="float32")
        self.early_feedback_ramp = numpy.linspace(0.5,0.1,num=early_feedback_loops,dtype="float32")
        self.early_highcut_filter = EffectFilter.CreateHighCutFilter(400)


    def applyreverb(self,float32_array_input):

        #Early Reflections
        float32_array_filtered=self.early_highcut_filter.applyfilter(float32_array_input)

        for counter in range(len(self.early_feedback_ramp)):
            processed_input = float32_array_filtered * self.early_feedback_ramp[counter]
            start_index = self.time_in_samples*(counter+1)
            end_index = (self.time_in_samples*(counter+1))+len(float32_array_input)
            self.delay_buffer[start_index:end_index] += processed_input

        float32_array_input += self.delay_buffer[0:len(float32_array_input)]
        self.delay_buffer = self.delay_buffer[len(float32_array_input):len(self.delay_buffer)]
        self.delay_buffer = numpy.append(self.delay_buffer,numpy.zeros(len(float32_array_input),dtype="float32"))

        return(float32_array_filtered)

    

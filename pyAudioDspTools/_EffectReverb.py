import numpy
from .config import chunk_size,sampling_rate
from .EffectFFTFilter import CreateLowCutFilter, CreateHighCutFilter

class CreateReverb:
    def __init__(self,time_in_ms=1500):
        self.reverb_time = time_in_ms
        self.reverb_time_in_samples = numpy.int((time_in_ms/1000)*sampling_rate)
        self.early_reflection_delay_1 = self.__CreateReverbDelayLine(self.reverb_time_in_samples,27,100,0,5000,False,True,True)
        self.early_reflection_delay_2 = self.__CreateReverbDelayLine(self.reverb_time_in_samples,127, 50, 0, 150,False,True,True)
        #self.early_reflection_delay_3 = self.__CreateReverbDelayLine(413, 1, 0, 600, False, True, True)
        #self.early_reflection_delay_4 = self.__CreateReverbDelayLine(1333, 1, 0, 2000, False, True, True)


    def applyreverb(self,float32_array_input):
        delay_line_1 = self.early_reflection_delay_1.apply(float32_array_input)
        delay_line_2 = self.early_reflection_delay_2.apply(float32_array_input)
        #delay_line_3 = self.early_reflection_delay_3.apply(float32_array_input)
        #delay_line_4 = self.early_reflection_delay_4.apply(float32_array_input)

        float32_array_input = delay_line_1 + delay_line_2 #+ delay_line_3 + delay_line_4 #+ float32_array_input

        return float32_array_input


    class __CreateReverbDelayLine:
        def __init__(self, reverb_time_in_samples,time_in_ms=500, feedback_loops=2, lowcut_filter_frequency=40, highcut_filter_frequency=12000,
                     use_lowcut_filter=False, use_highcut_filter=False, wet=False):
            self.time_in_samples = numpy.int(reverb_time_in_samples//feedback_loops)#numpy.int(time_in_ms * (sampling_rate / 1000))
            print(self.time_in_samples)
            self.wet = wet
            self.max_samples = reverb_time_in_samples #self.time_in_samples * (feedback_loops + 2)
            self.delay_buffer = numpy.zeros(int(self.max_samples), dtype="float32")
            self.feedback_ramp = numpy.linspace(0.3, 0.01, num=feedback_loops, dtype="float32")
            self.use_lowcut_filter = use_lowcut_filter
            self.use_highcut_filter = use_highcut_filter
            self.LowCutFilter = CreateLowCutFilter(lowcut_filter_frequency)
            self.HighcutFilter = CreateHighCutFilter(highcut_filter_frequency)

        def apply(self, float32_array_input):
            if (self.use_lowcut_filter == True):
                float32_array_input = self.LowCutFilter.apply(float32_array_input)
            if (self.use_highcut_filter == True):
                float32_array_input = self.HighcutFilter.apply(float32_array_input)

            for counter in range(len(self.feedback_ramp)-1):
                processed_input = float32_array_input * self.feedback_ramp[counter]
                start_index = self.time_in_samples * (counter + 1)
                end_index = (self.time_in_samples * (counter + 1)) + len(float32_array_input)
                self.delay_buffer[start_index:end_index] += processed_input

            if (self.wet == False):
                float32_array_input += self.delay_buffer[0:len(float32_array_input)]
            else:
                float32_array_input = self.delay_buffer[0:len(float32_array_input)]

            self.delay_buffer = self.delay_buffer[len(float32_array_input):len(self.delay_buffer)]
            self.delay_buffer = numpy.append(self.delay_buffer, numpy.zeros(len(float32_array_input), dtype="float32"))

            return (float32_array_input)

    def __CreateWhitenoise():
        whitenoise_time_array = numpy.arange(chunk_size)
        freqs = numpy.abs(numpy.fft.fftfreq(chunk_size, 1 / sampling_rate))
        f = numpy.zeros(chunk_size)
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
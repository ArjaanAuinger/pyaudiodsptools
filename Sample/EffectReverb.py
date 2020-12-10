import numpy
import EffectFilter
import EffectDelay

class CreateReverb:
    def __init__(self,time_in_ms=5000):
        self.early_reflection_delay_1 = EffectDelay.CreateDelay(17,2,0,100,False,True,True)
        self.early_reflection_delay_2 = EffectDelay.CreateDelay(29, 1, 0, 150,False,True,True)
        self.early_reflection_delay_3 = EffectDelay.CreateDelay(400, 1, 0, 600, False, True, True)
        self.early_reflection_delay_4 = EffectDelay.CreateDelay(1000, 1, 0, 2000, False, True, True)


    def applyreverb(self,float32_array_input):
        delay_line_1 = self.early_reflection_delay_1.applydelay(float32_array_input)
        delay_line_2 = self.early_reflection_delay_2.applydelay(float32_array_input)
        delay_line_3 = self.early_reflection_delay_3.applydelay(float32_array_input)
        delay_line_4 = self.early_reflection_delay_4.applydelay(float32_array_input)

        float32_array_input = delay_line_1 + delay_line_2 + delay_line_3 + delay_line_4 + float32_array_input

        return float32_array_input

    

import numpy

class CreateBitCrusher:

    def __init__(self):
        self.placeholder = True

    def apply(self,float_array_input):
        float_array_input = (float_array_input * 32767).astype('int16')
        float_array_input = float_array_input // 512
        float_array_input = float_array_input / 64
        return float_array_input
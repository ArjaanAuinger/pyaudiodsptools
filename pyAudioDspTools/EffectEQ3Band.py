import numpy
import copy

class CreateEQ3Band:
    def __init__(self,low_shelf_frequency,low_shelf_gain,mid_frequency,mid_gain,high_shelf_frequency,high_shelf_gain):
        self.LowdBgain = low_shelf_gain
        self.MiddBgain = mid_gain
        self.HighdBgain = high_shelf_gain
        self.Fs = 44100.0
        self.Pi = numpy.pi

        self.PrevChunkSampleLow = numpy.array([0.0,0.0])
        self.PrevOriginalChunkSampleLow = numpy.array([0.0,0.0,0.0])

        self.PrevChunkSampleMid = numpy.array([0.0,0.0])
        self.PrevOriginalChunkSampleMid = numpy.array([0.0,0.0,0.0])

        self.PrevChunkSampleHigh = numpy.array([0.0,0.0])
        self.PrevOriginalChunkSampleHigh = numpy.array([0.0,0.0,0.0])

        #Low Band Options and Coefficients
        self.LowShelfFreq = low_shelf_frequency
        self.LowShelfQ = 1.0;
        self.LowA = numpy.sqrt(10**(self.LowdBgain/20))
        self.LowShelfw0 = 2 * numpy.pi * self.LowShelfFreq / self.Fs
        self.LowShelfalpha = numpy.sin(self.LowShelfw0)/2 * numpy.sqrt((self.LowA + 1/self.LowA)*(1/self.LowShelfQ-1)+2)

        #Mid Band Options and Coefficients
        self.MidFreq = mid_frequency
        self.MidQ = 2.5;
        self.MidA = numpy.sqrt(10**(self.MiddBgain/20))
        self.Midw0 = 2 * numpy.pi * self.MidFreq / self.Fs
        self.Midalpha = numpy.sin(self.Midw0) / (2 * self.MidQ)

        #High Shelf Options ans Coefficients
        self.HighShelfFreq = high_shelf_frequency
        self.HighShelfQ = 1.0;
        self.HighA = numpy.sqrt(10**(self.HighdBgain/20))
        self.HighShelfw0 = 2 * numpy.pi * self.HighShelfFreq / self.Fs
        self.HighShelfalpha = numpy.sin(self.HighShelfw0)/2 * numpy.sqrt((self.HighA + 1/self.HighA)*(1/self.HighShelfQ-1)+2)

        #Low Shelf Coefficients calculation
        self.LOWb0 = self.LowA*((self.LowA+1) - (self.LowA-1)*numpy.cos(self.LowShelfw0) + 2*numpy.sqrt(self.LowA)*self.LowShelfalpha)
        self.LOWb1 = 2*self.LowA*((self.LowA-1) - (self.LowA+1)*numpy.cos(self.LowShelfw0))
        self.LOWb2 = self.LowA*((self.LowA+1) - (self.LowA-1)*numpy.cos(self.LowShelfw0) - 2*numpy.sqrt(self.LowA)*self.LowShelfalpha)
        self.LOWa0 = (self.LowA+1) + (self.LowA-1)*numpy.cos(self.LowShelfw0) + 2*numpy.sqrt(self.LowA)*self.LowShelfalpha
        self.LOWa1 = -2*((self.LowA-1) + (self.LowA+1)*numpy.cos(self.LowShelfw0))
        self.LOWa2 = (self.LowA+1) + (self.LowA-1)*numpy.cos(self.LowShelfw0) - 2*numpy.sqrt(self.LowA)*self.LowShelfalpha

        #Mid Coefficients calculation
        self.MIDb0 = 1 + self.Midalpha * self.MidA
        self.MIDb1 = -2 * numpy.cos(self.Midw0)
        self.MIDb2 = 1 - self.Midalpha * self.MidA
        self.MIDa0 = 1 + self.Midalpha / self.MidA
        self.MIDa1 = -2 * numpy.cos(self.Midw0)
        self.MIDa2 = 1 - self.Midalpha / self.MidA

        #High Shelf Coefficients calculation
        self.HIGHb0 = self.HighA*((self.HighA+1) + (self.HighA-1)*numpy.cos(self.HighShelfw0) + 2*numpy.sqrt(self.HighA)*self.HighShelfalpha)
        self.HIGHb1 = -2*self.HighA*((self.HighA-1) + (self.HighA+1)*numpy.cos(self.HighShelfw0))
        self.HIGHb2 = self.HighA*((self.HighA+1) + (self.HighA-1)*numpy.cos(self.HighShelfw0) - 2*numpy.sqrt(self.HighA)*self.HighShelfalpha)
        self.HIGHa0 = (self.HighA+1) - (self.HighA-1)*numpy.cos(self.HighShelfw0) + 2*numpy.sqrt(self.HighA)*self.HighShelfalpha
        self.HIGHa1 = 2*((self.HighA-1) - (self.HighA+1)*numpy.cos(self.HighShelfw0))
        self.HIGHa2 = (self.HighA+1) - (self.HighA-1)*numpy.cos(self.HighShelfw0) - 2*numpy.sqrt(self.HighA)*self.HighShelfalpha

    def applylowband (self,int_array_input):
        ILow = 2

        int_array_input_unprocessed = copy.deepcopy(int_array_input)
        int_array_input_unprocessed = numpy.insert(int_array_input_unprocessed,0,self.PrevOriginalChunkSampleLow)
        self.PrevOriginalChunkSampleLow = copy.deepcopy(int_array_input[-3:])
        int_array_input = numpy.insert(int_array_input,0,self.PrevChunkSampleLow)

        while (ILow < len(int_array_input)):
            int_array_input[ILow] = (self.LOWb0 / self.LOWa0 * int_array_input_unprocessed[ILow]) + (self.LOWb1 / self.LOWa0 * int_array_input_unprocessed[ILow-1]) + (self.LOWb2 / self.LOWa0 * int_array_input_unprocessed[ILow-2]) - (self.LOWa1 / self.LOWa0 * int_array_input[ILow - 1]) - (self.LOWa2 / self.LOWa0 * int_array_input[ILow - 2])
            ILow = ILow + 1 #increment the counter I by adding

        self.PrevChunkSampleLow = copy.deepcopy(int_array_input[-2:])
        int_array_input = int_array_input[2:]

        return int_array_input


    def applymidband(self, int_array_input):
        IMid = 2

        int_array_input_unprocessed = copy.deepcopy(int_array_input)
        int_array_input_unprocessed = numpy.insert(int_array_input_unprocessed, 0, self.PrevOriginalChunkSampleMid)
        self.PrevOriginalChunkSampleMid = copy.deepcopy(int_array_input[-3:])
        int_array_input = numpy.insert(int_array_input, 0, self.PrevChunkSampleMid)

        while (IMid < len(int_array_input)):
            int_array_input[IMid] = (self.MIDb0 / self.MIDa0 * int_array_input_unprocessed[IMid]) + (self.MIDb1 / self.MIDa0 * int_array_input_unprocessed[IMid-1]) + (self.MIDb2 / self.MIDa0 * int_array_input_unprocessed[IMid-2]) - (self.MIDa1 / self.MIDa0 * int_array_input[IMid - 1]) - (self.MIDa2 / self.MIDa0 * int_array_input[IMid - 2])
            IMid = IMid + 1  # increment the counter I by adding

        self.PrevChunkSampleMid = copy.deepcopy(int_array_input[-2:])
        int_array_input = int_array_input[2:]

        return int_array_input


    def applyhighband(self, int_array_input):
        IHigh = 2

        int_array_input_unprocessed = copy.deepcopy(int_array_input)
        int_array_input_unprocessed = numpy.insert(int_array_input_unprocessed, 0, self.PrevOriginalChunkSampleHigh)
        self.PrevOriginalChunkSampleHigh = copy.deepcopy(int_array_input[-3:])
        int_array_input = numpy.insert(int_array_input, 0, self.PrevChunkSampleHigh)

        while (IHigh < len(int_array_input)):
            int_array_input[IHigh] = (self.HIGHb0/self.HIGHa0 * int_array_input_unprocessed[IHigh]) + (self.HIGHb1 / self.HIGHa0 * int_array_input_unprocessed[IHigh-1]) + (self.HIGHb2 / self.HIGHa0 * int_array_input_unprocessed[IHigh-2]) - (self.HIGHa1 / self.HIGHa0 * int_array_input[IHigh - 1]) - (self.HIGHa2 / self.HIGHa0 * int_array_input[IHigh - 2])
            IHigh = IHigh + 1 #increment the counter I by adding

        self.PrevChunkSampleHigh = copy.deepcopy(int_array_input[-2:])
        int_array_input = int_array_input[2:]

        return int_array_input


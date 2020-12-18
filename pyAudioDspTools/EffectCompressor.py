import numpy
#import sys
#import math
#import array
from .config import sampling_rate, chunk_size

"""########################################################################################
Creating a Compressor class/device.
Init Parameters: 
    threshold_in_db: Defines when the compressor get active, decrease for more compression; negative [float] or [int]
    ratio: ratio/amount of compression in power; [float] between 0.0 and 1.0
    attack: attack of the compressor in milliseconds; [float] or [int]
    highshelf_db: release of the compressor in milliseconds; [float] 

applycompressor
    Applies the compressor to a 44100Hz/32 bit float numpy array of your choice.

This class introduces no latency.
###########################################################################################"""

class CreateCompressor:
    def __init__(self,threshold_in_db=-15,ratio=0.60,attack_in_ms=3.1,release_in_ms=30.1):
        self.ratio = ratio
        self.threshold_power = numpy.float32(10 ** (threshold_in_db / 20))
        self.attack_window = numpy.zeros(int((sampling_rate / 1000) * attack_in_ms),dtype="float32")
        self.attack_envelope = numpy.linspace(1.0,self.ratio,num=len(self.attack_window),dtype="float32")

        self.release_window = numpy.zeros(int((sampling_rate / 1000) * release_in_ms), dtype="float32")
        self.release_envelope = numpy.linspace(self.ratio,1.0, num=len(self.release_window), dtype="float32")
        self.counter_freeze = 0
        self.x = 0
        self.y = 0
        self.comp_state ="Resting"
        # x = attack envelope counter
        # y = release envelope counter


    def apply(self,int_array_input):
        int_array_input_bool_threshold = numpy.absolute(int_array_input) > self.threshold_power
        release_follow = False
        attack_follow = None
        release_break = None
        full_envelope = True
        counter_freeze = False
        freeze_params = False

        counter = 0
        x_max=len(self.attack_envelope)
        y_max=len(self.release_envelope)

        while counter < int(len(int_array_input_bool_threshold)):

            if int_array_input_bool_threshold[counter] == True or self.x != 0 or self.y != 0:
                if full_envelope == True and self.comp_state == "Resting":
                    self.x=0
                    self.comp_state="Attack"
                if full_envelope == False and self.comp_state =="Release":
                    self.x=x_max-int(self.y*(x_max/y_max))
                    counter_freeze = False
                    self.comp_state = "Attack"

                #Attack
                while self.x < x_max and self.comp_state =="Attack":
                    int_array_input[(counter)] = int_array_input[(counter)] * self.attack_envelope[self.x]
                    counter +=1
                    self.x +=1
                    if counter >= (len(int_array_input_bool_threshold)):
                        break
                if counter >= (len(int_array_input_bool_threshold)):
                    break

                #Hold
                while int_array_input_bool_threshold[counter] == True and self.comp_state =="Attack":
                    int_array_input[(counter)] = int_array_input[(counter)] * self.attack_envelope[x_max-1]
                    counter += 1
                    if counter >= (len(int_array_input_bool_threshold)):
                        break
                if counter >= (len(int_array_input_bool_threshold)):
                    break

                #Release
                self.comp_state = "Release"
                while self.y < y_max and self.comp_state=="Release":
                    self.x = 0
                    if int_array_input_bool_threshold[counter] == False:
                        int_array_input[(counter)] = int_array_input[(counter)] * self.release_envelope[self.y]
                        counter +=1
                        self.y+=1
                        if counter >= (len(int_array_input_bool_threshold)):
                            break

                    else:
                        if counter >= (len(int_array_input_bool_threshold)):
                            break
                        full_envelope = False
                        self.y=0
                        counter_freeze = True
                        break
                if counter >= (len(int_array_input_bool_threshold)):
                    break
                if self.y == y_max:
                    full_envelope = True
                    self.comp_state = "Resting"
                    self.x=0
                    self.y=0
            if counter_freeze == False:
                counter += 1
        return int_array_input

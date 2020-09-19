import numpy
import sys
import math
import array


class CreateGate:
    def __init__(self,threshold_in_db=-5,depth=0.1,attack=3.1,release=200.1):
        self.depth = depth
        self.threshold_power = numpy.float32(10 ** (threshold_in_db / 20))
        self.attack_window = numpy.zeros(int((44100 / 1000) * attack),dtype="float32")
        self.attack_envelope = numpy.linspace(1.0,1.0/self.depth,num=len(self.attack_window),dtype="float32")

        self.release_window = numpy.zeros(int((44100 / 1000) * release), dtype="float32")
        self.release_envelope = numpy.linspace(1.0/self.depth,1.0, num=len(self.release_window), dtype="float32")
        self.counter_freeze = 0
        self.x = 0
        self.y = 0
        self.comp_state ="Resting"
        print(self.attack_envelope)
        # x = attack envelope counter
        # y = release envelope counter


    def applygate(self,int_array_input):
        int_array_input_bool_threshold = numpy.absolute(int_array_input) > self.threshold_power
        int_array_input = int_array_input * self.depth
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













"""


            if int_array_input_bool_threshold[counter] == False and attack_follow == False:
                release_follow = True
                comp_state = "Release"

            if int_array_input_bool_threshold[counter] == True:
                if comp_state == "Release":
                    release_break == True
                attack_follow = True
                comp_state = "Attack"


            if attack_follow == True:
                if release_break == True:
                    x = int(y*(x_max/y_max))
                y = 0
                release_follow = False
                if x < x_max:
                    if int_array_input[counter] >= 0.0:
                        int_array_input[(counter)] = self.threshold_power + ((int_array_input[(counter)] - self.threshold_power)*self.attack_envelope[x])
                    else:
                        int_array_input[(counter)] = -self.threshold_power - ((-int_array_input[(counter)] - self.threshold_power) * self.attack_envelope[x])
                if x >= x_max:
                    if int_array_input_bool_threshold[(counter)] == True:
                        if int_array_input[counter] >= 0.0:
                            int_array_input[(counter)] = self.threshold_power + ((int_array_input[(counter)] - self.threshold_power) * self.attack_envelope[x_max-1])
                        else:
                            int_array_input[(counter)] = -self.threshold_power - ((-int_array_input[(counter)] - self.threshold_power) * self.attack_envelope[x_max-1])
                    else:
                        attack_follow = False
                        release_follow = True
                x += 1
                    #if int_array_input[counter] >= 0.0:
                        #int_array_input[counter] = self.threshold_power + ((int_array_input[(counter)] - self.threshold_power)*(1/self.ratio))
                    #else:
                        #int_array_input[counter] = -self.threshold_power - ((-int_array_input[(counter)] - self.threshold_power) * (1 / self.ratio))
                #x += 1
                #counter +=1

            if release_follow == True:
                x = 0
                if y < y_max:
                    print(int_array_input[(counter)])
                    if int_array_input[(counter)] >= 0.0:
                        int_array_input[(counter)] = self.threshold_power + ((int_array_input[(counter)] - self.threshold_power)*self.release_envelope[y])
                    else:
                        int_array_input[(counter)] = -self.threshold_power - ((-int_array_input[(counter)] - self.threshold_power) * self.release_envelope[y])
                if y >= y_max:
                    release_follow = False
                    comp_state = "Nothing"
                y +=1
                #counter +=1

        return int_array_input


"""






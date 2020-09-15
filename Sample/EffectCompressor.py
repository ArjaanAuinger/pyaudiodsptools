import numpy
import sys
import math
import array
"""
    def frame_count(ms=None):

        if ms is not None:
            return ms * (44100 / 1000.0)
        else:
            return float(len(seg) // frame_width)

    thresh_rms = 32767 * db_to_float(threshold)

    look_frames = int(frame_count(ms=attack))

    def rms_at(frame_i):
        return get_sample_slice(frame_i - look_frames, frame_i).rms

    def db_over_threshold(rms):
        if rms == 0: return 0.0
        db = ratio_to_db(rms / thresh_rms)
        return max(db, 0)

    output = []

    # amount to reduce the volume of the audio by (in dB)
    attenuation = 0.0

    attack_frames = frame_count(ms=attack)
    release_frames = frame_count(ms=release)
    for i in seg:
        rms_now = rms_at(i)

        # with a ratio of 4.0 this means the volume will exceed the threshold by
        # 1/4 the amount (of dB) that it would otherwise
        max_attenuation = (1 - (1.0 / ratio)) * db_over_threshold(rms_now)

        attenuation_inc = max_attenuation / attack_frames
        attenuation_dec = max_attenuation / release_frames

        if rms_now > thresh_rms and attenuation <= max_attenuation:
            attenuation += attenuation_inc
            attenuation = min(attenuation, max_attenuation)
        else:
            attenuation -= attenuation_dec
            attenuation = max(attenuation, 0)

        frame = seg.get_frame(i)
        if attenuation != 0.0:
            frame = audioop.mul(frame,
                                seg.sample_width,
                                db_to_float(-attenuation))

        output.append(frame)

    return seg._spawn(data=b''.join(output))
"""


class CreateCompressor:
    def __init__(self,threshold_in_db=-15,ratio=0.60,attack=0.1,release=0.1):
        self.ratio = ratio
        self.threshold_power = numpy.float32(10 ** (threshold_in_db / 20))
        self.attack_window = numpy.zeros(int((44100 / 1000) * attack),dtype="float32")
        self.attack_envelope = numpy.linspace(1.0,self.ratio,num=len(self.attack_window),dtype="float32")
        print(self.threshold_power)
        #print(self.attack_envelope)
        self.release_window = numpy.zeros(int((44100 / 1000) * release), dtype="float32")
        self.release_envelope = numpy.linspace(self.ratio,1.0, num=len(self.release_window), dtype="float32")
        self.counter_freeze = 0
        #print(self.attack_envelope)
        #print(self.release_envelope)
        #self.compression_envelope = numpy.ndarray()

    #def _createenvelope(self,bool_numpy_array_input,start=0):
        #for sample in bool_numpy_array_input:
            #if sample:

            #yield sample, bool_numpy_array_input
            #start += 1

    def applycompressor(self,int_array_input):
        compression_envelope = numpy.ones(len(int_array_input),dtype="float32")
        int_array_input = int_array_input.astype('float32')
        int_array_input = int_array_input/32768
        int_array_input_bool_threshold = numpy.absolute(int_array_input) > self.threshold_power
        release_follow = False
        attack_follow = None
        release_break = None
        full_envelope = True
        counter_freeze = False
        comp_state = "Nothing"

        counter = 0
        x=0
        y=0
        x_max=len(self.attack_envelope)
        y_max=len(self.release_envelope)

        while counter < int(len(int_array_input_bool_threshold)):
            #if counter < len(int_array_input):
                #counter += 1
            if counter == 0:
                counter += 1
                continue

            if counter == 1000:
                print("Stop")

            #x = attack envelope counter
            #y = release envelope counter
            if int_array_input_bool_threshold[counter] == True:
                if full_envelope == True:
                    x=0
                else:
                    x=x_max-int(y*(x_max/y_max))
                    counter_freeze = False
                #Attack
                while x < x_max:
                    if int_array_input[counter] >= 0.0:
                        int_array_input[(counter)] = int_array_input[(counter)] * self.attack_envelope[x]
                        counter +=1
                        x +=1
                    else:
                        int_array_input[(counter)] = int_array_input[(counter)] * self.attack_envelope[x]
                        counter +=1
                        x +=1
                #Absolut
                while int_array_input_bool_threshold[counter] == True:
                    if int_array_input[counter] >= 0.0:
                        int_array_input[(counter)] = int_array_input[(counter)] * self.attack_envelope[x_max-1]
                        counter += 1
                    else:
                        int_array_input[(counter)] = int_array_input[(counter)] * self.attack_envelope[x_max-1]
                        counter += 1
                #Release
                while y < y_max:
                    if int_array_input_bool_threshold[counter] == False:
                        if int_array_input[(counter)] >= 0.0:
                            int_array_input[(counter)] = int_array_input[(counter)] * self.release_envelope[y]
                            counter +=1
                            y+=1
                        else:
                            int_array_input[(counter)] = int_array_input[(counter)] * self.release_envelope[y]
                            counter +=1
                            y+=1
                    else:
                        full_envelope = False
                        y=0
                        counter_freeze = True
                        break
                if y == y_max:
                    full_envelope = True
                    x=0
                    y=0
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






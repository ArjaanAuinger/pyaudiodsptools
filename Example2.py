# Example 2: Creating a live audio stream and processing it by running the data though a lowcut filter.
# Is MONO.
# Has to be manually terminated in the IDE.

import pyaudio
import pyAudioDspTools
import time
import numpy
import sys

pyAudioDspTools.config.initialize(44100, 4096)

filterdevice = pyAudioDspTools.CreateLowCutFilter(300)


# Instantiate PyAudio
pyaudioinstance = pyaudio.PyAudio()

# The callback function first reads the current input and converts it to a numpy array, filters it and returns it.
def callback(in_data, frame_count, time_info, status):
    in_data = numpy.frombuffer(in_data, dtype=numpy.float32)
    in_data = filterdevice.apply(in_data)
    #print(numpydata)
    return (in_data, pyaudio.paContinue)


# The stream class of pyaudio. Setting all the variables, pretty self explanatory.
stream = pyaudioinstance.open(format=pyaudio.paFloat32,
                channels=1,
                rate=pyAudioDspTools.config.sampling_rate,
                input = True,
                output = True,
                frames_per_buffer = pyAudioDspTools.config.chunk_size,
                stream_callback = callback)

# start the stream
stream.start_stream()

# wait
while stream.is_active():
    time.sleep(5)
    print("Cpu load:", stream.get_cpu_load())

# stop stream
stream.stop_stream()
stream.close()

# close PyAudio
pyaudioinstance.terminate()
sys.exit()
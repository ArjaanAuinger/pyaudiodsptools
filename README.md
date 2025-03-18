![Logo](https://raw.githubusercontent.com/ArjaanAuinger/pyaudiodsptools/master/Logo.png)

[![Downloads](https://static.pepy.tech/personalized-badge/pyaudiodsptools?period=total&units=international_system&left_color=black&right_color=blue&left_text=Downloads)](https://pepy.tech/project/pyaudiodsptools)

pyAudioDspTools is a python 3 package for manipulating audio by just using numpy. This can be from a .wav or as a stream 
via pyAudio for example. pyAudioDspTool's only requirement is Numpy. The package is only a few kilobytes in size and 
well documented. You can find the readthedocs here: https://pyaudiodsptools.readthedocs.io/en/latest/

You can use pyAudioDspTools to start learning about audio dsp because all relevant operations are in plain sight,
no C or C++ code will be called and nearly no blackboxing takes place. You can also easily modify all available audio
effects and start writing your own, you only need to know python and numpy as well as audio dsp basics. As this package 
is released under the MIT licence, you can use it as you see fit. If you want to examine the code, just open the
'pyAudioDspTools' folder in this Git and you can see all the relevant modules.

# Quickstart

## Prerequesites
  - Python 3.6 or greater
  - Numpy (find here: https://numpy.org/)

## Installation with pip
To install pyAudioDspTools simply open a terminal in your venv and use pip:

  `pip install pyAudioDspTools`

The package is only a few kB in size, so it should download in an instant. After installing the package you can import it to your module in Python via:

  `import pyAudioDspTools`

pyAudioDspTools is device centered. Every sound effect processor is a class, imagine it being like an audio-device. You first create a class/device with certain settings and then run some numpy-arrays (which is your audio data) through them. This always follows a few simple steps, depending on if you want to modify data from a .wav file or realtime-stream. All classes are ready for realtime streaming and manage all relevant variables themselves.

If you have an RTX GPU and the the Cuda 11 or 12 toolkit installed, you can also install cupy for GPU processing of the FFT version of the filters as well as the 3 Band EQ:

  `pip install cupy-cuda11x` or `pip install cupy-cuda12x`

If you need more info you can visit the cupy website.


## Using pyAudioDspTools

Below you will find a few simple examples of processing your data. Example 1 will read a .wav
file, process the data and write it to a second .wav file. Example 2 will do the same, but with a stereo file. Example 3 will create a stream via the
pyAudio package and process everything in realtime. Example 4 is the same as Example 1, but using cupy and GPU acceleration.


### Example1.py : Processing from a .wav file.

```python

    import pyAudioDspTools

    pyAudioDspTools.config.initialize(44100, 4096)

    # Importing a mono .wav file and then splitting the resulting numpy-array in smaller chunks.
    full_data = pyAudioDspTools.Utility.MonoWavToNumpyFloat("TestFile16BitMono.wav")
    split_data = pyAudioDspTools.MakeChunks(full_data)


    # Creating the class/device, which is a lowcut filter
    filter_device = pyAudioDspTools.CreateLowCutFilter(800)


    # Setting a counter and process the chunks via filter_device.apply
    counter = 0
    for counter in range(len(split_data)):
        split_data[counter] = filter_device.apply(split_data[counter])
        counter += 1


    # Merging the numpy-array back into a single big one and write it to a .wav file.
    merged_data = pyAudioDspTools.CombineChunks(split_data)
    pyAudioDspTools.NumpyFloatToWav("output_audiofile.wav", merged_data)
```


### Example2.py : Stereo Processing from a .wav file.

```python

    import numpy
    import pyAudioDspTools

    pyAudioDspTools.config.initialize(44100, 4096)

    # Importing a mono .wav file and then splitting the resulting numpy-array in smaller chunks.
    left_channel, right_channel = pyAudioDspTools.Utility.StereoWavToNumpyFloat("TestFile16BitStereo.wav")
    split_data_left = pyAudioDspTools.MakeChunks(left_channel)
    split_data_right = pyAudioDspTools.MakeChunks(right_channel)


    # Creating the class/device, which is a lowcut filter
    filter_device_left = pyAudioDspTools.CreateLowCutFilter(800)
    filter_device_right = pyAudioDspTools.CreateLowCutFilter(800)


    # Setting a counter and process the chunks via filter_device.apply
    counter = 0
    for counter in range(len(split_data_left)):
        split_data_left[counter] = filter_device_left.apply(split_data_left[counter])
        split_data_right[counter] = filter_device_right.apply(split_data_right[counter])
        counter += 1


    # Merging the numpy-array back into a single big one and write it to a .wav file.
    merged_data_left = pyAudioDspTools.CombineChunks(split_data_left)
    merged_data_right = pyAudioDspTools.CombineChunks(split_data_right)

    pyAudioDspTools.NumpyFloatToWav("output_audiofile.wav",numpy.array([merged_data_left,merged_data_right]))
```

### Example3.py : Processing a live feed with pyaudio

```python

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
```

### Example4.py : Processing from a .wav file using cupy and GPU acceleration.
```python

    # Make sure you have cupy installed for this example. pyAudioDspTools will warn you if it cannot find the package.
    import pyAudioDspTools

    pyAudioDspTools.config.initialize(44100, 4096)

    # Importing a mono .wav file and then splitting the resulting numpy-array in smaller chunks.
    full_data = pyAudioDspTools.Utility.MonoWavToNumpyFloat("TestFile16BitMono.wav")
    split_data = pyAudioDspTools.MakeChunks(full_data)


    # Creating the class/device, which is a lowcut filter
    filter_device = pyAudioDspTools.CreateLowCutFilterGPU(800)


    # Setting a counter and process the chunks via filter_device.apply
    counter = 0
    for counter in range(len(split_data)):
        split_data[counter] = filter_device.apply(split_data[counter])
        counter += 1


    # Merging the numpy-array back into a single big one and write it to a .wav file.
    merged_data = pyAudioDspTools.CombineChunks(split_data)
    pyAudioDspTools.NumpyFloatToWav("output_audiofile.wav", merged_data)
```

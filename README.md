![alt text](https://raw.githubusercontent.com/ArjaanAuinger/pyaudiodsptools/master/Logo.png?token=ABCXOYG7HP6NFYXGLIRPGQ275XZYI?raw=true)

# pyAudioDspTools


pyAudioDspTools is a python 3 package for manipulating sound

# Quickstart

To install pyAudioDspTools simply open a terminal in your venv and use pip:

  `pip install pyAudioDspTools`

The package is only a few kB in size, so it should download in an instant. After installing the package you can import it to your module in Python via:

  `import pyAudioDspTools`

pyAudioDspTools is device centered. Every sound effect processor is a class, imagine it being like an audio-device. You first create a class/device with certain settings and then run some numpy-arrays (which is your audio data) through them. This always follows a few simple steps, depending on if you want to modify data from a .wav file or realtime-stream. All classes are ready for realtime streaming and manage all relevant variables themselves.

image of pipeline here


## Using pyAudioDspTools

Below you will find 2 simple examples of processing you data.Example 1 will read a .wav
file, process the data and write it to a second .wav file. Example 2 will create a stream via the
pyAudio package and process everything in realtime


### Processing from a .wav file.


    import pyAudioDspTools


    # Importing a mono .wav file and then splitting the resulting numpy-array in smaller chunks.
    full_data = pyAudioDspTools.MonoWavToNumpyFloat("some_path/your_audiofile.wav")
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
    NumpyFloatToWav("some_path/output_audiofile.wav", merged_data)`




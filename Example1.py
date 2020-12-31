import pyAudioDspTools

pyAudioDspTools.sampling_rate = 44100
pyAudioDspTools.chunk_size = 512

# Importing a mono .wav file and then splitting the resulting numpy-array in smaller chunks.
full_data = pyAudioDspTools.MonoWavToNumpyFloat("pyAudioDspTools/testmusic_mono.wav")
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
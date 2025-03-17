import numpy
import pyAudioDspTools

pyAudioDspTools.config.initialize(44100, 4096)

# Importing a stereo .wav file and then splitting the resulting numpy-array in smaller chunks.
left_channel, right_channel = pyAudioDspTools.Utility.StereoWavToNumpyFloat("TestFile16BitStereo.wav")
split_data_left = pyAudioDspTools.MakeChunks(left_channel)
split_data_right = pyAudioDspTools.MakeChunks(right_channel)


# Creating the classes/devices, which are lowcut filters
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

pyAudioDspTools.NumpyFloatToWav("output_audiofile.wav", numpy.array([merged_data_left,merged_data_right]))
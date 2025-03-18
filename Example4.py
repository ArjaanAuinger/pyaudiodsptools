# Make sure you have cupy installed for this example. pyAudioDspTools will warn you if it cannot find the package.
import pyAudioDspTools
import cupy

pyAudioDspTools.config.initialize(44100, 88200)

# Importing a mono .wav file and then splitting the resulting numpy-array in smaller chunks.
full_data = pyAudioDspTools.Utility.MonoWavToNumpyFloat("TestFile16BitMono.wav")
split_data = cupy.array(pyAudioDspTools.MakeChunks(full_data))


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
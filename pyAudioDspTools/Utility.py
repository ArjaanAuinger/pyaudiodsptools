import numpy
import sys
import wave
import math
import struct
import copy
from . import config
def MakeChunks(float32_array_input):
    """Converts a long numpy array in multiple small ones for processing

    Parameters
    ----------
    float_array_input : float
        The array, which you want to slice.

    Returns
    -------
    numpy array
        The sliced arrays.

    """
    number_of_chunks = math.ceil(numpy.float32(len(float32_array_input)/config.chunk_size))
    if len(float32_array_input) % number_of_chunks != 0:
        samples_to_append = config.chunk_size - (len(float32_array_input) % config.chunk_size)
        #print(number_of_chunks)
        float32_array_input = numpy.append(float32_array_input,numpy.zeros(samples_to_append,dtype="float32"))
    float32_chunked_array = numpy.split(float32_array_input, number_of_chunks)
    return float32_chunked_array


def CombineChunks(float_array_input):
    """Converts a sliced array back into one long one. Use this if you want to write to .wav

    Parameters
    ----------
    float_array_input : float
        The array, which you want to slice.

    Returns
    -------
    numpy array
        The sliced arrays.

    """
    float32_array_output = numpy.array([],dtype="float32")
    for chunk in float_array_input:
        float32_array_output = numpy.append(float32_array_output,chunk)
    return float32_array_output


def MixSignals(*args):
    """Adds several numpy arrays. Used for mixing audio signals

    Parameters
    ----------
    args : 1D numpy-arrays
        Multiple arrays.

    Returns
    -------
    1D numpy array
        A single array.

    """
    mixed_signal = numpy.zeros(len(args[0]))
    for signal in args:
        try:
            mixed_signal = mixed_signal + signal
        except:
            raise Exception("Something went wrong. Make sure, that the Numpy arrays are equal in length.")
    mixed_signal = numpy.clip(mixed_signal, -1.0, 1.0)
    return mixed_signal

"""######Converts 16-bit to dBV (+-1.0 Volt)######"""
def ConvertdBVTo16Bit(float_array_input):
    float_array_input = numpy.clip(float_array_input, -1.0, 1.0)
    float_array_output = numpy.int16(float_array_input * (2 ** 15 - 1))
    return float_array_output

"""######Converts dBV (+-1.0 Volt) to 16-bit######"""
def Convert16BitTodBV(int_array_input):
    float_array_output = numpy.float32(int_array_input/32767)
    return float_array_output

"""######Converts 16-bit signed integer to 8-bit signed integer######"""
def Dither16BitTo8Bit(int_array_input):
    rectangular_dither_array = numpy.random.randint(-1, 1, size=int_array_input.size)
    int_array_dithered = numpy.around(int_array_input / 256, decimals=0).astype('int16')

    int_array_dithered = numpy.add(int_array_dithered, rectangular_dither_array)
    int_array_dithered = numpy.clip(int_array_dithered, a_min=-127, a_max=127)
    int_array_dithered.astype('int8')
    # int_array_output = (int_array_dithered*256).astype('int16')
    return int_array_dithered

"""######Converts 32-bit signed integer to 16-bit signed integer######"""
def Dither32BitIntTo16BitInt(int_array_input):
    rectangular_dither_array = numpy.random.randint(-1, 1, size=int_array_input.size)
    int_array_dithered = numpy.around(int_array_input / 65535, decimals=0).astype('int32')

    int_array_dithered = numpy.add(int_array_dithered, rectangular_dither_array)
    int_array_dithered = numpy.clip(int_array_dithered, a_min=-32767, a_max=32767)
    int_array_dithered = int_array_dithered.astype('int16')
    # int_array_output = (int_array_dithered*65535).astype('int32')
    return int_array_dithered

"""
#BSD Licence
#Not Tested!
def Import24BitWavTo16Bit(wav_file,data):
    if sampwidth != 3:
        print("wav_file is not 24-Bit! Cannot perform operation.")
        return
    else:
        a = numpy.empty((num_samples, nchannels, 4), dtype=numpy.uint8)
        raw_bytes = numpy.fromstring(data, dtype=numpy.uint8)
        a[:, :, :sampwidth] = raw_bytes.reshape(-1, nchannels, sampwidth)
        a[:, :, sampwidth:] = (a[:, :, sampwidth - 1:sampwidth] >> 7) * 255
        result = a.view('<i4').reshape(a.shape[:-1])
        return result
"""
def InfodBV(float_array_input):
    """Prints the average sum as decibel whereas 1.0 is 0dB.

    Parameters
    ----------
    float_array_input : float
        The audio data.

    Returns
    -------
    dBV : float
        Average power in dB.

    """
    count = 0
    added_sample_value = 0.0
    average_sample_value = 0.0
    audio_array = copy.deepcopy(float_array_input)
    audio_array = numpy.where(audio_array > 0, audio_array, audio_array * (-1))
    average_sample_value = audio_array.sum()/audio_array.size
    amplitude = average_sample_value/1
    dBV = 20 * math.log10(amplitude)
    return dBV

def InfodBV16Bit(int_array_input):
    """Prints the average sum as decibel whereas 32767 is 0dB.

    Parameters
    ----------
    int_array_input : int
        The audio data.

    Returns
    -------
    dB16 : float
        Average power in dB.

    """
    count = 0
    added_sample_value = 0.0
    average_sample_value = 0.0
    audio_array = copy.deepcopy(int_array_input)
    audio_array = numpy.where(audio_array > 0, audio_array, audio_array * (-1))
    average_sample_value = audio_array.sum()/audio_array.size
    amplitude = average_sample_value/32767
    dB16 = 20 * math.log10(amplitude)
    return dB16


def VolumeChange(float_array_input, gain_change_in_db,overflow_protection = True):
    """Increases or decreses the volume of a signal in decibel.

    Parameters
    ----------
    float_array_input : float
        The array, which you want to be processed.
    gain_change_in_db : float
        The amount of change in volume in decibel.
    overflow_protection : bool
        If true it will clip every value above 1.0 and below -1.0 to 1.0 and -1.0

    Returns
    -------
    numpy array
        The processed array

    """
    float_array_input = (10 ** (gain_change_in_db/20))*float_array_input

    if (overflow_protection == True):
        float_array_input = numpy.clip(float_array_input, -1.0, 1.0)

    return float_array_input


def MonoWavToNumpy16BitInt(wav_file_path):
    """Imports a .wav file as a numpy array. All values will be scaled to be
    between -32768 and 32767.

    Parameters
    ----------
    wav_file_path : string
        Follows the normal python path rules.

    Returns
    -------
    numpy array : int16
        The imported array

    """
    wav_file = wave.open(wav_file_path)
    samples = wav_file.getnframes()
    audio = wav_file.readframes(samples)
    audio_as_numpy_array = numpy.frombuffer(audio, dtype=numpy.int16)
    return audio_as_numpy_array

def MonoWavToNumpyFloat(wav_file_path):
    """Imports a .wav file as a numpy array. All values will be scaled to be
    between -1.0 and 1.0 for further processing.

    Parameters
    ----------
    wav_file_path : string
        Follows the normal python path rules.

    Returns
    -------
    numpy array : float
        The imported array

    """
    wav_file = wave.open(wav_file_path)
    samples = wav_file.getnframes()
    audio = wav_file.readframes(samples)
    audio_as_numpy_array = numpy.frombuffer(audio, dtype=numpy.int16)
    audio_as_numpy_array = (audio_as_numpy_array.astype('float32')/32768)
    return audio_as_numpy_array


def StereoWavToNumpyFloat(wav_file_path):
    """Imports a stereo .wav file as a numpy array.
    All values will be scaled to be between -1.0 and 1.0 for further processing.

    Parameters
    ----------
    wav_file_path : string
        Follows the normal python path rules.

    Returns
    -------
    numpy array : float
        The imported array with shape (n_samples, 2) for stereo files.
    """
    with wave.open(wav_file_path, 'rb') as wav_file:
        n_channels = wav_file.getnchannels()
        if n_channels != 2:
            raise ValueError("This function supports only stereo .wav files.")

        samples = wav_file.getnframes()
        audio = wav_file.readframes(samples)

        # Convert audio to numpy array
        audio_as_numpy_array = numpy.frombuffer(audio, dtype=numpy.int16)

        # Reshape for stereo audio
        audio_as_numpy_array = audio_as_numpy_array.reshape(-1, 2)

        # Normalize the audio
        audio_as_numpy_array = audio_as_numpy_array.astype('float32') / 32768

        # Split the audio into two separate arrays
        left_channel = audio_as_numpy_array[:, 0]
        right_channel = audio_as_numpy_array[:, 1]

        return left_channel, right_channel

def NumpyFloatToWav(wav_file_path, numpy_array):
    """
    Exports a numpy array as a .wav file.

    Parameters
    ----------
    numpy_array : numpy array
        The array containing audio data. Must be of shape (n_samples,) for mono
        or (n_samples, 2) for stereo.

    wav_file_path : string
        The path where the .wav file will be saved.

    sample_rate : int, optional
        The sample rate of the audio. Default is 44100 Hz.
    """
    # Check if stereo and reshape if necessary ro (n_samples, 2) instead of (2, n_samples) in case array is [left_channel, right_channel]
    if numpy_array.ndim == 2 and numpy_array.shape[0] == 2:
        numpy_array = numpy_array.T  # Transpose to (n_samples, 2)

    # Determine number of channels
    n_channels = 1 if numpy_array.ndim == 1 else numpy_array.shape[1]

    # Ensure the audio data is in the range [-1.0, 1.0]
    if not numpy.any((numpy_array >= -1) & (numpy_array <= 1)):
        raise ValueError("Array values should be in the range [-1.0, 1.0]")

    # Convert float array to int16
    int_data = (numpy_array * 32767).astype('int16')

    with wave.open(wav_file_path, 'wb') as wav_file:
        wav_file.setnchannels(n_channels)
        wav_file.setsampwidth(2)  # 2 bytes for 'int16'
        wav_file.setframerate(config.sampling_rate)
        wav_file.writeframes(int_data.tobytes())
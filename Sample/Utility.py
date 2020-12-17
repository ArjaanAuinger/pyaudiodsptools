import numpy
import sys
import wave
import math
import struct
import copy
import config

"""######Converts a long numpy array in multiple small ones for processing#####"""
def MakeChunks(float32_array_input,chunk_size=config.chunk_size):
    number_of_chunks = math.ceil(numpy.float32(len(float32_array_input)/chunk_size))
    if len(float32_array_input) % number_of_chunks != 0:
        samples_to_append = chunk_size - (len(float32_array_input) % chunk_size)
        #print(number_of_chunks)
        float32_array_input = numpy.append(float32_array_input,numpy.zeros(samples_to_append,dtype="float32"))
    float32_chunked_array = numpy.split(float32_array_input, number_of_chunks)
    return float32_chunked_array

"""######Converts multiple numpy arrays into one long array for writing to a .wav file#####"""
def CombineChunks(numpy_array_input):
    float32_array_output = numpy.array([],dtype="float32")
    for chunk in numpy_array_input:
        float32_array_output = numpy.append(float32_array_output,chunk)
    return float32_array_output


"""######Adds several numpy arrays. Used for mixing audio signals#######"""
def MixSignals(*args):
    mixed_signal = numpy.zeros(len(args[0]))
    for signal in args:
        try:
            mixed_signal = mixed_signal + signal
        except:
            raise Exception("Something went wrong. Make sure, that the Numpy arrays are equal in length.")
    mixed_signal = numpy.clip(mixed_signal, -1.0, 1.0)
    return mixed_signal

"""######Converts dBu (+-1.736 Volt) to 16-bit. Good for SPICE automation testing#######"""
def ConvertdBuTo16Bit(float_array_input):
    float_array_input = numpy.where(float_array_input < 1.736, float_array_input, 1.736)
    float_array_input = numpy.where(float_array_input > -1.736, float_array_input, -1.736)
    float_array_output = numpy.int16(float_array_input * ((2 ** 15 - 1)/1.736))
    return float_array_output

"""######Converts 16-bit to dBu (+-1.736 Volt). Good for SPICE automation testing#######"""
def Convert16BitTodBu(int_array_input):
    float_array_output = numpy.float32((int_array_input/32767)*1.736)
    return float_array_output

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
    count = 0
    added_sample_value = 0.0
    average_sample_value = 0.0
    audio_array = copy.deepcopy(int_array_input)
    audio_array = numpy.where(audio_array > 0, audio_array, audio_array * (-1))
    average_sample_value = audio_array.sum()/audio_array.size
    amplitude = average_sample_value/32767
    dB16 = 20 * math.log10(amplitude)
    return dB16


def VolumeChange(float32_array_input, gain_change_in_db):
    float32_array_input = (10 ** (gain_change_in_db/20))*float32_array_input
    float32_array_input = numpy.clip(float32_array_input, -1.0, 1.0)
    return float32_array_input


def MonoWavToNumpy16BitInt(wav_file_path):
    wav_file = wave.open(wav_file_path)
    samples = wav_file.getnframes()
    audio = wav_file.readframes(samples)
    audio_as_numpy_array = numpy.frombuffer(audio, dtype=numpy.int16)
    return audio_as_numpy_array

def MonoWavToNumpy32BitFloat(wav_file_path):
    wav_file = wave.open(wav_file_path)
    samples = wav_file.getnframes()
    audio = wav_file.readframes(samples)
    audio_as_numpy_array = numpy.frombuffer(audio, dtype=numpy.int16)
    audio_as_numpy_array = (audio_as_numpy_array.astype('float32')/32768)
    return audio_as_numpy_array

def Numpy16BitIntToMonoWav44kHz(filename, data):
    """
    Write a numpy array as a WAV file

    Parameters
    ----------
    filename : string or open file handle
        Output wav file
    data : ndarray
        A 1-D or 2-D numpy array of either integer or float data-type.

    Notes
    -----
    * The file can be an open file or a filename.

    * Writes a simple uncompressed WAV file.
    * The bits-per-sample will be determined by the data-type.
    * To write multiple-channels, use a 2-D array of shape
      (Nsamples, Nchannels).

    """
    if hasattr(filename,'write'):
        fid = filename
    else:
        fid = open(filename, 'wb')

    try:
        dkind = data.dtype.kind
        if not (dkind == 'i' or dkind == 'f' or (dkind == 'u' and data.dtype.itemsize == 1)):
            raise ValueError("Unsupported data type '%s'" % data.dtype)

        fid.write(b'RIFF')
        fid.write(b'\x00\x00\x00\x00')
        fid.write(b'WAVE')
        # fmt chunk
        fid.write(b'fmt ')
        if dkind == 'f':
            comp = 3
        else:
            comp = 1
        if data.ndim == 1:
            noc = 1
        else:
            noc = data.shape[1]
        bits = data.dtype.itemsize * 8
        sbytes = config.sampling_rate*(bits // 8)*noc
        ba = noc * (bits // 8)
        fid.write(struct.pack('<ihHIIHH', 16, comp, noc, config.sampling_rate, sbytes, ba, bits))
        # data chunk
        fid.write(b'data')
        fid.write(struct.pack('<i', data.nbytes))
        if data.dtype.byteorder == '>' or (data.dtype.byteorder == '=' and sys.byteorder == 'big'):
            data = data.byteswap()
        _array_tofile(fid, data)

        # Determine file size and place it in correct
        #  position at start of the file.
        size = fid.tell()
        fid.seek(4)
        fid.write(struct.pack('<i', size-8))

    finally:
        if not hasattr(filename,'write'):
            fid.close()
        else:
            fid.seek(0)
    return

if sys.version_info[0] >= 3:
    def _array_tofile(fid, data):
        # ravel gives a c-contiguous buffer
        fid.write(data.ravel().view('b').data)
else:
    def _array_tofile(fid, data):
        fid.write(data.tostring())


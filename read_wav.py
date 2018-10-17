import os
import wave
import numpy as np
import struct

def bit24_2_32(sub_bytes):
    if sub_bytes[2] < 128:
        return sub_bytes + b'\x00'
    else:
        return sub_bytes + b'\xff'


def read_wav_file(file_path):
    wave_file = wave.open(file_path, 'rb')
    params = wave_file.getparams()
    channels, sampleWidth, frameRate, frames = params[:4]
    data_bytes = wave_file.readframes(frames)  # 读取音频，字符串格式
    wave_file.close()

    wave_data = np.zeros(channels * frames)
    if sampleWidth == 2:
        wave_data = np.fromstring(data_bytes, dtype=np.int16)  # 将字符串转化为int
    elif sampleWidth == 3:
        samples = np.zeros(channels * frames)
        for i in np.arange(samples.size):
            sub_bytes = data_bytes[i * 3:(i * 3 + 3)]
            sub_bytes = bit24_2_32(sub_bytes)
            samples[i] = struct.unpack('i', sub_bytes)[0]
        wave_data = samples

    wave_data = wave_data.astype(np.float)
    wave_data = np.reshape(wave_data, [frames, channels])

    return wave_data, frameRate

def list_wav_files(root_path):
    list = []
    for filename in os.listdir(root_path):
        pathname = os.path.join(root_path, filename)
        if os.path.isfile(pathname):
            (shotname, extension) = os.path.splitext(filename)
            if extension == '.wav':
                list = list + [pathname]

        else:
            list = list + list_wav_files(pathname)

    return list
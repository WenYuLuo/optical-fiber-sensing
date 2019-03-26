from scipy import signal
import wave
import numpy as np
import os
import struct


def bit24_2_32(sub_bytes):
    if sub_bytes[2] < 128:
        return sub_bytes + b'\x00'
    else:
        return sub_bytes + b'\xff'


def read_wav_file(file_name):
    """
    读wav文件
    :param file_name: string, filename
    :return: data = wave_data, sample rate = frameRate 
    """
    wave_file = wave.open(file_name, 'rb')
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


def find_ext_files(root_path, ext='.wav'):
    """
    列出root_path下文件拓展名为ext的所有文件
    :param root_path: string, root path of the files
    :param ext: string, None or '.wav', '.txt'... (defult='.wav') 
    :return: list of files with specific extension name  
    """
    list = []
    for filename in os.listdir(root_path):
        pathname = os.path.join(root_path, filename)
        if os.path.isfile(pathname):
            (shotname, extension) = os.path.splitext(filename)
            if extension == ext:
                list = list + [pathname]

        else:
            list = list + find_ext_files(pathname)

    return list


def list_files(root_path):
    """
    列出root_path下所有文件夹
    :param root_path: string, root path
    :return: list of folders
    """
    list_folder = []
    for filename in os.listdir(root_path):
        pathname = os.path.join(root_path, filename)
        if os.path.isfile(pathname):
            continue
        list_folder = list_folder + [pathname]
    return list_folder


def mkdir(path):
    """
    创建文件夹
    :param path: string, 目标文件夹名（带路径）
    :return: None
    """
    # 去除首位空格
    path = path.strip()
    # 去除尾部 / 符号
    path = path.rstrip("/")
    # 判断路径是否存在
    if not os.path.exists(path):
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)


def resample(input_signal, src_fs, tar_fs):
    """
    信号重采样
    :param input_signal:输入信号
    :param src_fs:输入信号采样率
    :param tar_fs:输出信号采样率
    :return:输出信号
    """
    if src_fs == tar_fs:
        return input_signal

    dtype = input_signal.dtype
    audio_len = len(input_signal)
    audio_time_max = 1.0 * (audio_len - 1) / src_fs
    src_time = 1.0 * np.linspace(0, audio_len, audio_len, endpoint=False) / src_fs
    tar_time = 1.0 * np.linspace(0, np.int(audio_time_max * tar_fs+0.5), np.int(audio_time_max * tar_fs+0.5), endpoint=False) / tar_fs
    output_signal = np.interp(tar_time, src_time, input_signal).astype(dtype)
    return output_signal


def butter_worth_filter(audio, cutoff, fs, btype='lowpass', N=8):
    """
    :param audio: 输入信号
    :param cutoff: array_like 截止频率 
    :param fs: 采样率
    :param btype: 滤波器类型（默认‘lowpass’），{‘lowpass’, ‘highpass’, ‘bandpass’, ‘bandstop’}
    :param N: 滤波器阶数，默认为8
    :return: 滤波信号
    """
    wn = 2 * cutoff / fs
    b, a = signal.butter(N, wn, btype)
    audio_filted = signal.filtfilt(b, a, audio)
    return audio_filted


def enframe(signal, nw, inc, winfunc=None):

    '''
    将音频信号转化为帧
    参数含义：
    signal:原始音频型号
    nw:每一帧的长度(这里指采样点的长度，即采样频率乘以时间间隔)
    inc:相邻帧的间隔（同上定义）
    winfunc:加窗函数， 默认不加窗
    :return   frames： 分帧结果
            last_valid：最后一帧的有效长度
    '''

    signal_length = len(signal)
    if signal_length <= nw:             # 若信号长度小于一个帧的长度，则帧数定义为1
        nf = 1
    else:   # 否则，计算帧的总长度
        # 取最大能够截取的帧数，不足舍去
        nf = int(np.ceil((1.0*signal_length - nw + inc)/inc))   # nf为帧数

    # 填充
    pad_length = int((nf-1)*inc+nw)                 # 所有帧加起来总的铺平后的长度
    zeros = np.zeros((pad_length-signal_length,))   # 不够的长度使用0填补，类似于FFT中的扩充数组操作
    pad_signal = np.concatenate((signal, zeros))    # 填补后的信号记为pad_signal

    # 对所有帧的时间点进行抽取，得到 nf*nw 长度的矩阵
    indices = np.tile(np.arange(0, nw), (nf, 1)) + np.tile(np.arange(0, nf * inc, inc), (nw, 1)).T
    indices = np.array(indices, dtype=np.int32)     # 将indices转化为矩阵
    frames = pad_signal[indices]        # 得到帧信号
    last_valid = nw - (pad_length - signal_length)
    if winfunc is None:
        return frames, last_valid
    win = np.tile(winfunc, (nf, 1))     # window窗函数，这里默认取1
    return frames * win, last_valid   # 返回帧信号矩阵


def energy_normalize(xs):
    energy = np.sqrt(np.sum(xs ** 2, 1))
    for i in range(energy.shape[0]):
        xs[i] /= energy[i]
    return xs
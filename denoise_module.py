from ctypes import *
import os
import read_wav
import webrtcvad
vad = webrtcvad.Vad(3)
vad.is_speech()

if __name__ == '__main__':


    libpath = './libwebDenoise.so'
    lib_denoise = cdll.LoadLibrary(libpath)

    dict = {0: '', 1: '', 2: '', 3: ''}
    dict[0] = "/media/fish/Elements/Project/光纤传感/光纤音频/布放光缆"
    dict[1] = "/media/fish/Elements/Project/光纤传感/光纤音频/机械施工"
    dict[2] = "/media/fish/Elements/Project/光纤传感/光纤音频/井内人工动作"
    dict[3] = "/media/fish/Elements/Project/光纤传感/光纤音频/雨水流入井内冲击光缆"

    # 去噪音频存储路径
    dict_denoised = {0: '', 1: '', 2: '', 3: ''}
    dict_denoised[0] = "/media/fish/Elements/Project/光纤传感/光纤音频_去噪/布放光缆"
    dict_denoised[1] = "/media/fish/Elements/Project/光纤传感/光纤音频_去噪/机械施工"
    dict_denoised[2] = "/media/fish/Elements/Project/光纤传感/光纤音频_去噪/井内人工动作"
    dict_denoised[3] = "/media/fish/Elements/Project/光纤传感/光纤音频_去噪/雨水流入井内冲击光缆"

    # # 10.9 数据
    # dict = {0: '', 1: '', 2: '', 3: '', 4: ''}
    # dict[0] = "/media/fish/Elements/Project/光纤传感/2018.10.9音频样本/线路1/人工井内施工"
    # dict[1] = "/media/fish/Elements/Project/光纤传感/2018.10.9音频样本/线路1/下雨告警"
    # dict[2] = "/media/fish/Elements/Project/光纤传感/2018.10.9音频样本/线路2/放缆"
    # dict[3] = "/media/fish/Elements/Project/光纤传感/2018.10.9音频样本/线路2/人工井内施工"
    # dict[4] = "/media/fish/Elements/Project/光纤传感/2018.10.9音频样本/线路2/下雨"
    #
    #
    # dict_denoised = {0: '', 1: '', 2: '', 3: '', 4: ''}
    # dict_denoised[0] = "/media/fish/Elements/Project/光纤传感/2018.10.9音频样本_去噪/线路1/人工井内施工"
    # dict_denoised[1] = "/media/fish/Elements/Project/光纤传感/2018.10.9音频样本_去噪/线路1/下雨告警"
    # dict_denoised[2] = "/media/fish/Elements/Project/光纤传感/2018.10.9音频样本_去噪/线路2/放缆"
    # dict_denoised[3] = "/media/fish/Elements/Project/光纤传感/2018.10.9音频样本_去噪/线路2/人工井内施工"
    # dict_denoised[4] = "/media/fish/Elements/Project/光纤传感/2018.10.9音频样本_去噪/线路2/下雨"



    for key in dict:
        print(dict[key])
        print(dict_denoised[key])
        if not os.path.exists(dict_denoised[key]):
            os.makedirs(dict_denoised[key])
        wav_files = read_wav.list_wav_files(dict[key])
        print('count:', len(wav_files))
        for pathname in wav_files:
            [path, wavname_ext] = os.path.split(pathname)
            # print(pathname)
            wavname = wavname_ext.split('.')[0]
            out_pathname = os.path.join(dict_denoised[key], wavname+'_out.wav')
            in_file = c_char_p(pathname.encode('utf-8'))
            out_file = c_char_p(out_pathname.encode('utf-8'))
            lib_denoise.noise_suppression(in_file, out_file)
            print(' ')

    # in_file = c_char_p(b'/media/fish/Elements/Project/光纤传感/光纤音频/布放光缆/碰触加穿缆_杂有轻微小型机械声/20170111095756.wav')
    # out_file = c_char_p(b'out.wav')
    # lib_denoise.noise_suppression(in_file, out_file)


import sys
sys.path.append("..")
from hmm_optical_sensing import *
import common
import numpy as np
import json


def load_audio(path, seg, nw, n_mfcc):
    # seg = 4 # 2s分段
    # nw = 1024 # 帧长约23ms*4
    # n_mfcc = 32  # mfcc 维数

    folder_list = common.list_files(path)
    data_dict = {}
    for folder in folder_list:
        species = folder.split('\\')[-1]
        wav_list = common.find_ext_files(folder, '.wav')
        instance = dataObejct()
        for wav in wav_list:
            wavname = wav.split('\\')[-1]
            wav_data, fs = common.read_wav_file(wav)
            # 多通道仅取一通道
            if wav_data.ndim > 1:
                wav_data = wav_data[:, 0]
            wav_data = wav_data.T
            ref_value = 2 ** 12 - 1
            wav_data = wav_data / ref_value  # wave幅值归一化

            filter_data = common.butter_worth_filter(wav_data, cutoff=1000, fs=fs, btype='high', N=8)

            seg_mfcc, frame_num = enframe_and_feature_extract(filter_data, seg, nw, fs, n_mfcc)

            # # 输出为[n_mfcc, n_sample]
            # mfcc_data = librosa.feature.mfcc(y=filter_data, sr=fs, n_mfcc=n_mfcc, n_fft=nw, hop_length=inc)

            instance.append(audio_name=wavname, origin=filter_data, frame=seg_mfcc, frame_num=frame_num)
        data_dict[species] = instance
    return data_dict


def main():
    """
    :param arg:
     输入文件路径：
        文件树结构（例如）：
            ——arg
                ——人工                    （此目录下的wav文件均会被读取，并将标记为人工井内施工）
                    20180310253.wav
                    ——2018.1.3
                        20180103111.wav
                        
                ——机械施工
                ——放缆
            
    :return: 
    """
    with open('.\\test_config.json', encoding='UTF-8') as file:
        test_config = json.load(file, strict=False)
    path = test_config['test_data_path']
    seg_param = test_config['seg_param']
    seg = seg_param['seg']  # 4s分段
    nw = seg_param['nw']   # 帧长约23ms*4
    n_mfcc = seg_param['n_mfcc']   # mfcc 维数
    save_file = test_config['hmm_model']
    data_dict = load_audio(path, seg, nw, n_mfcc)

    # save_file = '..\\model'
    hmms_model = hmms()
    model_list = common.find_ext_files(save_file, ext='.npy')
    for model_file in model_list:
        hmms_model.load_one_model(model_file)  # 加载模型
    # hmms_model.load_model(save_file)
    model_num = len(hmms_model.hmms)

    for key in data_dict:
        print('当前预测音频文件夹：%s' % key)
        instance = data_dict[key]
        audio_num = instance.get_num()
        species_count = np.zeros(model_num)
        for j in range(audio_num):
            # print('\t音频名：%s' % instance.audio_name[j])
            frame_data = instance.frame[j]
            length = instance.frame_num[j]
            predicts = hmms_model.batch_predict(frame_data, length=length)
            count = np.bincount(predicts)
            # for i in range(len(count)):
            #     print('\t\t%s:%d' % (hmms_model.model_name[i], count[i]), end='\t')

            major = np.argmax(count)
            if hmms_model.model_name[major] == '背景音':
                count[major] = 0
            major = np.argmax(count)
            species_count[major] += 1
            # print('\n\t\t预测结果：%s\n' % hmms_model.model_name[major])
        print('当前种类识别分布：')
        species_count /= audio_num
        for i in range(model_num):
            print('\t%s: %f' % (hmms_model.model_name[i], species_count[i]))


if __name__ == '__main__':
    # main('G:\\Project\\光纤传感\\2018.10.9音频样本\\线路1')
    # main('G:\\Project\\光纤传感\\2018.10.9音频样本\\线路2')
    # main('G:\\Project\\光纤传感\\音频集合')
    main()


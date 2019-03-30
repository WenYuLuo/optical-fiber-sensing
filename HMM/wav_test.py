from hmm_optical_sensing import *
import common


def load_audio(path):
    seg = 2 # 2s分段
    nw = 2048 # 帧长约23ms*4
    n_mfcc = 32  # mfcc 维数

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


def main(arg):
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
    data_dict = load_audio(arg)

    save_file = 'hmms_model_200.npy'
    hmms_model = hmms()
    hmms_model.load_model(save_file)

    for key in data_dict:
        print('当前预测音频种类：%s' % key)
        instance = data_dict[key]
        audio_num = instance.get_num()
        for j in range(audio_num):
            print('\t音频名：%s' % instance.audio_name[j])
            frame_data = instance.frame[j]
            length = instance.frame_num[j]
            predicts = hmms_model.batch_predict(frame_data, length=length)
            count = np.bincount(predicts)
            for i in range(len(count)):
                print('\t\t%s:%d' % (hmms_model.model_name[i], count[i]), end='\t')

            major = np.argmax(count)
            if hmms_model.model_name[major] == '背景音':
                count[major] = 0
            major = np.argmax(count)
            print('\n\t\t预测结果：%s\n' % hmms_model.model_name[major])


if __name__ == '__main__':
    # main('G:\\Project\\光纤传感\\2018.10.9音频样本\\线路1')
    # main('G:\\Project\\光纤传感\\2018.10.9音频样本\\线路2')
    main('G:\\Project\\光纤传感\\音频集合')


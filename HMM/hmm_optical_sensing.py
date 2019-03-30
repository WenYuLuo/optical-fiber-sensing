import librosa
from hmmlearn import hmm
import common
import random
import numpy as np
import os


def load_optical_data(path, seg, nw, n_mfcc):
    """
    :param path: 音频存储路径
    :param nw: 帧长
    :param n_mfcc: mfcc特征维数
    :return: 
    train_dict，dict {{'frame':分帧数据 ,'origin':原始音频数据（滤波）, 'frame_num':帧数},}
    test_dict，dict {{'frame':分帧数据 ,'origin':原始音频数据（滤波）, 'frame_num':帧数},}
    """
    list_folders = common.list_files(path)
    train_list = []
    test_list = []
    max_frame_num_train = 0
    max_frame_index = -1
    train_frame_num_list = []

    for i in range(len(list_folders)):
        name = list_folders[i].split('\\')[-1]
        list_wavs = common.find_ext_files(list_folders[i], ext='.wav')
        print('%d = %s num：%d' % (i, list_folders[i], len(list_wavs)))

        list_wavs = shuffle_list(list_wavs)
        instance = dataObejct()
        instance.set_name(name)
        debug = 0
        for wavname in list_wavs:
            if debug >= 200:
                break
            debug += 1
            wav_data, fs = common.read_wav_file(wavname)
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

        split_ = int(instance.get_num() / 2)
        shuffled_instance = instance.shuffle()
        train_samples = shuffled_instance[split_:]
        # 确定训练集最大帧数，留作样本平衡使用。
        train_samples.recompute_total()
        train_samples.set_name(name)
        max_frame = train_samples.total_frame
        train_frame_num_list.append(max_frame)
        if max_frame > max_frame_num_train:
            max_frame_num_train = max_frame
            max_frame_index = i
        test_samples = shuffled_instance[:split_]
        test_samples.set_name(name)

        train_list.append(train_samples)
        test_list.append(test_samples)

    if max_frame_index == -1:
        raise ValueError('the max frame index is not updated!(for data balance)')

    print('最大帧数：%d' % train_frame_num_list[max_frame_index])

    # 样本平衡
    for i in range(len(list_folders)):
        if i == max_frame_index:
            continue

        # 抽取样本，添加随机offset(0,256),(256,512)分帧
        instance = dataObejct()
        ind = 0
        total_audio = train_list[i].get_num()
        while train_frame_num_list[i] < max_frame_num_train:
            index = ind % total_audio
            samples = train_list[i].origin[index]
            offset = random.randrange(1, nw * 2)
            new_seg_mfcc, new_frame_num = enframe_and_feature_extract(samples[offset:], seg, nw, fs, n_mfcc)
            instance.append(audio_name=train_list[i].audio_name[index], origin=samples[offset:], frame=new_seg_mfcc, frame_num=new_frame_num)
            train_frame_num_list[i] += sum(new_frame_num)
            ind += 1

        train_list[i].appends(instance)

    return train_list, test_list


# def shuffle_data(data_dict):
#     temp = [i for i in range(len(data_dict['frame_num']))]
#     random.shuffle(temp)
#     origin = []
#     frame = []
#     frame_num = []
#     for i in temp:
#         origin.append(data_dict['origin'][i])
#         frame.append(data_dict['frame'][i])
#         frame_num.append(data_dict['frame_num'][i])
#     shuffled = {'frame': frame, 'origin': origin, 'frame_num': frame_num}
#     return shuffled

def shuffle_list(list):
    temp = [i for i in range(len(list))]
    random.shuffle(temp)
    shuffled = [list[i] for i in temp]
    return shuffled


def enframe_and_feature_extract(data, seg, nw, fs, n_mfcc):
    seg_length = seg * fs
    frames, last_valid = common.enframe(data, seg_length, int(3 * seg_length / 4))

    # 能量归一化
    frames = common.energy_normalize(frames)

    # 帧移
    inc = int(nw / 2)
    seg_mfcc = np.empty((0, n_mfcc))
    frame_num = []
    for i in range(frames.shape[0]):
        data = frames[i]
        if i == frames.shape[0]-1:
            if last_valid < 4 * nw:
                break
            data = data[:last_valid]
        mfcc_data = librosa.feature.mfcc(y=data, sr=fs, n_mfcc=n_mfcc + 1, n_fft=nw, hop_length=inc)
        seg_mfcc = np.vstack((seg_mfcc, mfcc_data[1:n_mfcc + 1].T))
        frame_num.append(mfcc_data.shape[1])
    return seg_mfcc, frame_num


class dataObejct:
    def __init__(self, audio_name=None, origin=None, frame=None, frame_num=None):
        self.name = None
        self.audio_name = []
        self.origin = []
        self.frame = []
        self.frame_num = []
        self.total_frame = 0
        if origin is not None:
            if isinstance(origin, list):
                self.audio_name = audio_name
                self.origin = origin
                self.frame = frame
                self.frame_num = frame_num
                for seg_frame in self.frame_num:
                    self.total_frame += sum(seg_frame)
            else:
                self.append(audio_name, origin, frame, frame_num)

    def set_name(self, name):
        self.name = name

    def append(self, audio_name, origin, frame, frame_num):
        # if len(origin) == 1:
        self.audio_name.append(audio_name)
        self.origin.append(origin)
        self.frame.append(frame)
        self.frame_num.append(frame_num)
        self.total_frame += sum(frame_num)
        # else:
        #     self.origin += origin
        #     self.frame += frame
        #     self.frame_num += frame_num
        #     for seg_frame in frame_num:
        #         self.total_frame += sum(seg_frame)

    def appends(self, dataObejct_value):
        self.audio_name += dataObejct_value.audio_name
        self.origin += dataObejct_value.origin
        self.frame += dataObejct_value.frame
        self.frame_num += dataObejct_value.frame_num
        for seg_frame in dataObejct_value.frame_num:
            self.total_frame += sum(seg_frame)

    def shuffle(self):
        temp = [i for i in range(len(self.origin))]
        random.shuffle(temp)
        shuffled = dataObejct()
        for i in temp:
            shuffled.append(self.audio_name[i], self.origin[i], self.frame[i], self.frame_num[i])
        return shuffled

    def get_num(self):
        return len(self.origin)

    def recompute_total(self):
        self.total_frame = 0
        for seg_frame in self.frame_num:
            self.total_frame += sum(seg_frame)

    def __getitem__(self, index):
        # temp = dataObejct()
        return dataObejct(self.audio_name[index], self.origin[index], self.frame[index], self.frame_num[index])

    def __setitem__(self, index, dataObejct_value):
        self.audio_name[index] = dataObejct_value.audio_name
        self.origin[index] = dataObejct_value.origin
        self.frame[index] = dataObejct_value.frame
        self.frame_num[index] = dataObejct_value.frame_num


class hmms:
    def __init__(self, n_components=0, n_mixs=0, n_iter=0):
        self.n_components = n_components
        self.n_mixs = n_mixs
        self.n_iter = n_iter
        self.hmms = []
        self.model_name = []
        # self.startprob = [1 / n_components] * n_components

    def train(self, train_list, length_list, name_list=None):
        """
        :param train_list: list, 训练数据列表
        :param length_list: 序列长度列表
        :param name_list: 各hmm模型名称， 默认为None
        :return: hmms： list, hmm模型list
        
        comment: 模型个数由train_list的个数类别决定
        """
        self.model_name = name_list
        print('开始训练hmm模型！')
        for i in range(len(train_list)):
            if name_list is not None:
                print('\t训练%s模型...'%name_list[i])
            model = hmm.GMMHMM(n_components=self.n_components,
                               n_mix=self.n_mixs,
                               covariance_type="diag")
            model.n_iter = self.n_iter
            model.fit(train_list[i], length_list[i])
            # # 重置初始概率
            # model.startprob_ = self.startprob
            self.hmms.append(model)
        print('所有模型训练完成！')

    def predict(self, x):
        max_index = -1
        max_score = None
        for i in range(len(self.hmms)):
            model_score = self.hmms[i].score(x)
            if (max_score is None) or (max_score < model_score):
                max_score = model_score
                max_index = i
        return max_index

    def batch_predict(self, xs, length):
        scores = []
        for i in range(len(self.hmms)):
            model_score = self.hmms[i].score(xs, lengths=length)
            scores.append(model_score)
        scores = np.array(scores)
        predicts = np.argmax(scores, axis=0)
        return predicts

    def save_model(self, filename):
        x = filename.split('.')
        if x[-1] != 'npy':
            raise ValueError('the extension of  filename should be npy!')
        hmm_list = []
        model_num = len(self.hmms)
        for i in range(model_num):
            hmm = self.hmms[i]
            hmm_dict = {}
            hmm_dict['n_components'] = hmm.n_components
            hmm_dict['n_mix'] = hmm.n_mix
            hmm_dict['covariance_type'] = hmm.covariance_type
            hmm_dict['startprob'] = hmm.startprob_
            hmm_dict['transmat'] = hmm.transmat_
            hmm_dict['means'] = hmm.means_
            hmm_dict['covars'] = hmm.covars_
            hmm_dict['weights'] = hmm.weights_
            hmm_dict['model_name'] = self.model_name[i]
            hmm_list.append(hmm_dict)
        np.save(filename, hmm_list)

    def load_model(self, filename):
        hmm_list = np.load(filename)
        num_model = len(hmm_list)
        for i in range(num_model):
            hmm_dict = hmm_list[i]
            model = hmm.GMMHMM(n_components=hmm_dict['n_components'],
                               n_mix=hmm_dict['n_mix'],
                               covariance_type=hmm_dict['covariance_type'])
            model.startprob_ = hmm_dict['startprob']
            model.transmat_ = hmm_dict['transmat']
            model.means_ = hmm_dict['means']
            model.covars_ = hmm_dict['covars']
            model.weights_ = hmm_dict['weights']
            self.hmms.append(model)
            self.model_name.append(hmm_dict['model_name'])


def reshape_data(data_list, n_mfcc):
    # frame_dict = {}
    # length_dict = {}
    frame_list = []
    length_list = []
    name_list = []
    for i in range(len(data_list)):
        instance = data_list[i]
        frames = np.empty((0, n_mfcc))
        frames_num = []
        for j in range(len(instance.origin)):
            frame_data = instance.frame[j]
            frame_data = frame_data.reshape((-1, n_mfcc))
            frames = np.vstack((frames, frame_data))
            frames_num += instance.frame_num[j]
        # frame_dict[instance.name] = frames
        # length_dict[instance.name] = frames_num
        name_list.append(instance.name)
        frame_list.append(frames)
        length_list.append(frames_num)
    return frame_list, length_list, name_list


def main():
    path = 'E:\DailyResearch\Project\杨老师\gmm_hmm\剪辑素材'
    seg = 2 # 2s分段
    nw = 2048 # 帧长约23ms*4
    n_mfcc = 32  # mfcc 维数
    train_list, test_list = load_optical_data(path, seg=seg, nw=nw, n_mfcc=n_mfcc)

    frame_list, length_list, name_list = reshape_data(train_list, n_mfcc)

    # 创建并初始化hmm模型
    n_components = 4
    n_mixs = 4
    n_iter = 1000
    hmm_models = hmms(n_components=n_components, n_mixs=n_mixs, n_iter=n_iter)
    hmm_models.train(frame_list, length_list, name_list)

    save_file = 'hmms_model_200.npy'
    hmm_models.save_model(save_file)

    # train accuracy
    print('train accuracy:')
    for i in range(len(frame_list)):
        length = length_list[i]
        data = frame_list[i]
        predicts = hmm_models.batch_predict(data, length=length)
        where_correct = np.where(predicts == i)[0]
        correct_num = where_correct.shape[0]
        acc = correct_num/len(length) * 100
        print('\tthe %s class: %f ' %(name_list[i], acc))


    # # test accuracy
    # frame_list, length_list, name_list = reshape_data(test_list, n_mfcc)
    # print('test accuracy:')
    # for i in range(len(frame_list)):
    #     length = length_list[i]
    #     data = frame_list[i]
    #     predicts = hmm_models.batch_predict(data, length=length)
    #     where_correct = np.where(predicts == i)[0]
    #     correct_num = where_correct.shape[0]
    #     acc = correct_num / len(length) * 100
    #     print('\tthe %s class: %f ' % (name_list[i], acc))

    class_num = len(test_list)
    audio_matrix = np.zeros((class_num, class_num))
    for i in range(class_num):
        seg_correct = 0
        audio_correct = 0
        seg_num = 0
        instance = test_list[i]
        audio_num = instance.get_num()
        for j in range(audio_num):
            frame_data = instance.frame[j]
            frame_data = frame_data.reshape((-1, n_mfcc))
            length = instance.frame_num[j]
            seg_num += len(length)
            predicts = hmm_models.batch_predict(frame_data, length=length)
            where_correct = np.where(predicts == i)[0]
            seg_correct += where_correct.shape[0]
            # major = sorted([(np.sum(predicts == i), i) for i in set(predicts)])
            count = np.bincount(predicts)
            major = np.argmax(count)
            audio_matrix[i, major] += 1
            if i == major:
                audio_correct += 1
        seg_acc = seg_correct / seg_num * 100
        audio_acc = audio_correct / audio_num
        print('\tthe %s class:' % name_list[i])
        print('\t\tsegment acc:', seg_acc)
        print('\t\taudio acc:', audio_acc)
    print('音频识别混淆矩阵：')
    print(audio_matrix)


if __name__ == '__main__':
    main()
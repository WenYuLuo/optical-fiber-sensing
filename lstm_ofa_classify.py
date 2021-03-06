import numpy as np
import tensorflow as tf
import time
import read_wav
from scipy import signal
import matplotlib.pyplot as plt
import webrtcvad



'''
光纤振动拾音器音频分类

'''

def enframe(signal, nw, inc, winfunc):

    '''
    将音频信号转化为帧
    参数含义：
    signal:原始音频型号
    nw:每一帧的长度(这里指采样点的长度，即采样频率乘以时间间隔)
    inc:相邻帧的间隔（同上定义）
    '''

    signal_length = len(signal)         # 信号总长度 441000
    if signal_length <= nw:             # 若信号长度小于一个帧的长度，则帧数定义为1
        nf = 1
    else:   # 否则，计算帧的总长度
        nf = int(np.ceil((1.0*signal_length - nw + inc)/inc))   # nf=1722

    pad_length = int((nf-1)*inc+nw)                 # 所有帧加起来总的铺平后的长度 441088
    zeros = np.zeros((pad_length-signal_length,))   # 不够的长度使用0填补，类似于FFT中的扩充数组操作
    pad_signal = np.concatenate((signal, zeros))    # 填补后的信号记为pad_signal 441088

    # 对所有帧的时间点进行抽取，得到 nf*nw 长度的矩阵
    indices = np.tile(np.arange(0, nw), (nf, 1)) + np.tile(np.arange(0, nf * inc, inc), (nw, 1)).T
    indices = np.array(indices, dtype=np.int32)     # 将indices转化为矩阵 (1722，512)
    frames = pad_signal[indices]        # 得到帧信号
    win = np.tile(winfunc, (nf, 1))     # window窗函数，这里默认取1
    return frames * win   # 返回帧信号矩阵


class LSTMcell(object):
    def __init__(self, incoming, D_input, D_cell, initializer, f_bias=1.0):
        # var
        # the shape of incoming is [n_samples, n_steps, D_cell]
        self.incoming = incoming
        self.D_input = D_input
        self.D_cell = D_cell
        # parameters
        # igate = W_xi.* x + W_hi.* h + b_i
        self.W_xi = initializer([self.D_input, self.D_cell])
        self.W_hi = initializer([self.D_cell, self.D_cell])
        self.b_i = tf.Variable(tf.zeros([self.D_cell]))
        # fgate = W_xf.* x + W_hf.* h + b_f
        self.W_xf = initializer([self.D_input, self.D_cell])
        self.W_hf = initializer([self.D_cell, self.D_cell])
        self.b_f = tf.Variable(tf.constant(f_bias, shape=[self.D_cell]))
        # ogate = W_xo.* x + W_ho.* h + b_o
        self.W_xo = initializer([self.D_input, self.D_cell])
        self.W_ho = initializer([self.D_cell, self.D_cell])
        self.b_o = tf.Variable(tf.zeros([self.D_cell]))
        # cell = W_xc.* x + W_hc.* h + b_c
        self.W_xc = initializer([self.D_input, self.D_cell])
        self.W_hc = initializer([self.D_cell, self.D_cell])
        self.b_c = tf.Variable(tf.zeros([self.D_cell]))

        # init cell and hidden state whose shapes are [n_samples, D_cell]
        init_for_both = tf.matmul(self.incoming[:, 0, :], tf.zeros([self.D_input, self.D_cell]))
        self.hid_init = init_for_both
        self.cell_init = init_for_both
        # because tf.scan only takes two arguments, the hidden state and cell are needed to merge
        self.previous_h_c_tuple = tf.stack([self.hid_init, self.cell_init])
        # transpose the tensor so that the first dim is time_step
        self.incoming = tf.transpose(self.incoming, perm=[1, 0, 2])

    def one_step(self, previous_h_c_tuple, current_x):
        # to split hidden state and cell
        prev_h, prev_c = tf.unstack(previous_h_c_tuple)

        # computing
        # input gate
        i = tf.sigmoid(
            tf.matmul(current_x, self.W_xi) +
            tf.matmul(prev_h, self.W_hi) +
            self.b_i)
        # forget Gate
        f = tf.sigmoid(
            tf.matmul(current_x, self.W_xf) +
            tf.matmul(prev_h, self.W_hf) +
            self.b_f)
        # output Gate
        o = tf.sigmoid(
            tf.matmul(current_x, self.W_xo) +
            tf.matmul(prev_h, self.W_ho) +
            self.b_o)
        # new cell info
        c = tf.tanh(
            tf.matmul(current_x, self.W_xc) +
            tf.matmul(prev_h, self.W_hc) +
            self.b_c)
        # current cell
        current_c = f * prev_c + i * c
        # current hidden state
        current_h = o * tf.tanh(current_c)

        return tf.stack([current_h, current_c])

    def all_steps(self):
        # inputs shape : [n_sample, n_steps, D_input]
        # outputs shape : [n_steps, n_sample, D_output]
        hstates = tf.scan(fn=self.one_step,
                          elems=self.incoming,
                          initializer=self.previous_h_c_tuple,
                          name='hstates')[:, 0, :, :]
        return hstates


# LSTM init and func
def weight_init(shape):
    initial = tf.random_uniform(shape, minval=-np.sqrt(5) * np.sqrt(1.0 / shape[0]),
                                maxval=np.sqrt(5) * np.sqrt(1.0 / shape[0]))
    return tf.Variable(initial, trainable=True)


def zero_init(shape):
    initial = tf.Variable(tf.zeros(shape))
    return tf.Variable(initial, trainable=True)


def orthogonal_initializer(shape, scale=1.0):
    scale = 1.0
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)  # this needs to be corrected to float32
    return tf.Variable(scale * q[:shape[0], :shape[1]], trainable=True, dtype=tf.float32)


def bias_init(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def shufflelists(data):
    ri = np.random.permutation(len(data))
    data = [data[i] for i in ri]
    return data


def highpass(xn, fs, fl=1000):
    if xn.ndim > 1:
        xn = xn[:, 0]

    Wn = fl / fs  # Convert 3 - dB frequency
    b, a = signal.butter(5, Wn, 'highpass')
    xn = signal.filtfilt(b, a, xn)
    return xn


def vad_process(data):
    data_len = 1024
    length = len(data)
    energy_arr = []
    i = 0
    while True:
        if (i+1)*data_len > length:
            break
        clip = data[i*data_len:(i+1)*data_len]
        energy = np.sqrt(np.sum(clip**2))
        energy_arr.append(energy)
        i += 1
    energy_arr = np.array(energy_arr)
    energy_mean = np.mean(energy_arr)
    # energy_std = np.std(energy_arr)
    threshold = energy_mean * 0.8

    # fake_fs = 8000 # 假设采样8khz
    # duration = 30 # 时长30ms
    # data_len = int(fake_fs * duration / 1000)
    # stream_list = pcm_stream(data, data_len) # pcm 分帧
    # is_slience = [0] * len(stream_list)
    # vad = webrtcvad.Vad(2)
    # for i in range(len(stream_list)):
    #     is_active = vad.is_speech(stream_list[i], fake_fs)
    #     is_slience[i] = 0 if is_active else 1

    index_active = np.where(energy_arr > threshold)[0]
    active_list = []
    # is_begin = False
    # start_index = index_active[0]
    # for i in range(len(index_active)):
    #     if not is_begin:
    #         start_index = index_active[i]
    #         is_begin = True
    #     if i+1 >= len(index_active):
    #         end_index = index_active[i]
    #         active_list.append((start_index*data_len, (end_index+1)*data_len))
    #         break
    #     if index_active[i+1] - index_active[i] > 1:
    #         end_index = index_active[i]
    #         active_list.append((start_index*data_len, (end_index+1)*data_len))
    #         is_begin = False
    #     else:
    #         continue

    for i in range(len(index_active)):
        start_index = index_active[i]
        # 第一帧和最后一帧不计入
        if start_index == 0 or start_index >= len(energy_arr) - 1:
            continue
        active_list.append((start_index * data_len - 512, (start_index + 1) * data_len + 512))

    return active_list


def pcm_stream(data, data_len):
    stream = []
    length = len(data)
    i = 0
    while True:
        if (i+1)*data_len > length:
            break
        clip = data[i*data_len:(i+1)*data_len]*100
        clip = clip.astype(np.int16)
        clip = clip.tobytes()
        stream.append(clip)
        i += 1
    return stream


def data_enframe(data, label_key):
    nw = 512
    inc = 256
    win_fun = np.hamming(nw)
    frames = enframe(data, nw, inc, win_fun)  # (1722，512) 1722帧，每帧长度512，每帧间隔长度256

    frames = np.fft.fft(frames)
    frames = np.sqrt(frames.real ** 2 + frames.imag ** 2)

    frames = frames[:, :int(nw/2)]

    # frames = energy_normalize(frames)

    frames = np.expand_dims(np.expand_dims(frames, axis=0), axis=0)
    frames = list(frames)

    label = [0, 0, 0, 0]
    label[label_key] = 1
    label = np.array([[label]])
    label = list(label)
    sample = frames + label
    return sample


def energy_normalize(xs):
    energy = np.sqrt(np.sum(xs ** 2, 1))
    for i in range(energy.shape[0]):
        xs[i] /= energy[i]
    return xs

# In[4]:
if __name__ == '__main__':

    # d bict = {0: '', 1: '', 2: '', 3: ''}
    # dict[0] = "/media/fish/Elements/Project/光纤传感/光纤音频/布放光缆"
    # dict[1] = "/media/fish/Elements/Project/光纤传感/光纤音频/机械施工"
    # dict[2] = "/media/fish/Elements/Project/光纤传感/光纤音频/井内人工动作"
    # dict[3] = "/media/fish/Elements/Project/光纤传感/光纤音频/雨水流入井内冲击光缆"

    # 去噪音频存储路径
    dict = {0: '', 1: '', 2: '', 3: ''}
    dict[0] = "/media/fish/Elements/Project/光纤传感/光纤音频_去噪/布放光缆"
    dict[1] = "/media/fish/Elements/Project/光纤传感/光纤音频_去噪/机械施工"
    dict[2] = "/media/fish/Elements/Project/光纤传感/光纤音频_去噪/井内人工动作"
    dict[3] = "/media/fish/Elements/Project/光纤传感/光纤音频_去噪/雨水流入井内冲击光缆"
    # dict[4] = "/media/fish/Elements/Project/光纤传感/光纤音频_去噪/静音"
    # slience_label = 4

    train = [] #384
    test = [] #96

    num_clips = 25000

    # slience_train = []
    # slience_test = []
    #
    # slience_count = 0

    for key in dict:
        current_train = []
        current_test = []

        count = 0
        if dict[key] == '':
            continue
        # key = 2
        print(dict[key])
        wav_files = read_wav.list_wav_files(dict[key])

        for pathname in wav_files:

            # # balance
            # if count > 30000:
            #     break

            wave_data, frameRate = read_wav.read_wav_file(pathname)

            wave_data = highpass(wave_data, frameRate, fl=1000) # 高通滤波(若为多通道仅使用第一通道数据)

            # if key == 3 or 4:
            #     active_pos_list = [[0, len(wave_data)]]
            # else:
            #     active_pos_list = vad_process(wave_data)

            active_pos_list = vad_process(wave_data)

            wave_data = wave_data.T

            wava_mean = float(np.sqrt(np.sum(wave_data**2))/len(wave_data))

            ref_value = (2 ** 12 - 1) / wava_mean
            wave_data = wave_data / ref_value  # wave幅值归一化

            # plt.plot(wave_data)
            # plt.show()
            detected_visual = np.zeros_like(wave_data)
            for i in range(len(active_pos_list)):
                # if pos[0] != 0:
                #

                # if pos[1] - pos[0] < 1024:
                #     continue
                pos = active_pos_list[i]
                detected_visual[pos[0]:pos[1]] = 1
                clip_data = wave_data[pos[0]:pos[1]]
                clip_data = energy_normalize(clip_data.reshape((1, 2048)))[0]
                sample = data_enframe(clip_data, key) # [段数,帧数,数据长度]
                count += 1
                if count % 5 == 0:
                    current_test.append(sample)
                else:
                    current_train.append(sample)
                # if i == 0 and pos[0] != 0:
                #     # 起始静音段
                #     slience_start = 0
                #     slience_end = pos[0]
                #     clip_data = wave_data[slience_start:slience_end]
                #     sample = data_enframe(clip_data, slience_label)
                #     slience_count += sample[0].shape[1]
                #     if slience_count % 5 == 0:
                #         slience_test.append(sample)
                #     else:
                #         slience_train.append(sample)
                #
                # if i+1 < len(active_pos_list):
                #     # 静音段
                #     slience_start = pos[1]
                #     slience_end = active_pos_list[i+1][0]
                #     clip_data = wave_data[slience_start:slience_end]
                #     sample = data_enframe(clip_data, slience_label)
                #     slience_count += sample[0].shape[1]
                #     if slience_count % 5 == 0:
                #         slience_test.append(sample)
                #     else:
                #         slience_train.append(sample)

            # import pylab as pl
            # time = np.arange(0, wave_data.shape[0])
            # verbose = True
            # if verbose:
            #     pl.subplot(211)
            #     spec = wave_data.tolist()
            #     pl.specgram(spec, Fs=22050, scale_by_freq=True, sides='default')
            #     pl.subplot(212)
            #     pl.plot(time, wave_data)
            #     pl.plot(time, detected_visual)
            #     pl.title('high pass filter')
            #     pl.xlabel('time')
            #     pl.show()
        print('实际提取段数：', count)
        current_train = shufflelists(current_train)
        current_test = shufflelists(current_test)
        train_len = len(current_train)
        test_len = len(current_test)
        while count < num_clips:
            if count % 5 == 0:
                index = count % test_len
                temp = current_test[index]
                current_test.append(temp)
            else:
                index = count % train_len
                temp = current_train[index]
                current_train.append(temp)
            count += 1
        else:
            current_train = current_train[:int(num_clips * 4 / 5)]
            current_test = current_train[:int(num_clips / 5)]
        print('重复采样至 %d 段数.' % (len(current_test)+len(current_train)))
        train += current_train
        test += current_test

    # slience_train = shufflelists(slience_train)
    # slience_to_be_used_count = 0
    # index = 0
    # while True:
    #     slience_to_be_used_count += slience_train[index][0].shape[1]
    #     if slience_to_be_used_count > 20000:
    #         break
    #     index += 1
    # slience_train = slience_train[:index]
    # slience_test = shufflelists(slience_test)[:100]
    # train += slience_train
    # test += slience_test


    # print('总体静音帧数:', slience_count)
    # print('训练使用静音帧数:', slience_to_be_used_count)
    print('num of train sequences:%s' %len(train))  # 384
    print('num of test sequences:%s' %len(test))    # 96
    print('shape of inputs:', test[0][0].shape)     # (1,1722,512)
    print('shape of labels:', test[0][1].shape)     # (1,4)

    D_input = 256
    D_label = 4
    learning_rate = 1e-4
    num_units = 256

    inputs = tf.placeholder(tf.float32, [None, None, D_input], name="inputs")
    labels = tf.placeholder(tf.float32, [None, D_label], name="labels")

    rnn_cell = LSTMcell(inputs, D_input, num_units, orthogonal_initializer)
    rnn_out = rnn_cell.all_steps()
    # reshape for output layer
    rnn = tf.reshape(rnn_out, [-1, num_units])
    # output layer
    W = weight_init([num_units, D_label])
    b = bias_init([D_label])
    output = tf.nn.softmax(tf.matmul(rnn, W) + b)

    loss = -tf.reduce_mean(labels*tf.log(output))

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    saver = tf.train.Saver()

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # 训练并记录
    def train_epoch(epoch):
        for k in range(epoch):
            train_shift = shufflelists(train)
            accumulated_acc = 0
            for i in range(len(train)):
                train_acc, m_ = sess.run((accuracy, train_step), feed_dict={inputs: train_shift[i][0], labels: train_shift[i][1]})
                # print(train_loss)
                accumulated_acc += train_acc
            print(k, 'train:', accumulated_acc / len(train))
            # tl = 0
            # dl = 0
            # for i in range(len(test)):
            #     dl += sess.run(loss, feed_dict={inputs: test[i][0], labels: test[i][1]})
            # for i in range(len(train)):
            #     tl += sess.run(loss, feed_dict={inputs: train[i][0], labels: train[i][1]})
            # print(k, 'train:', round(tl/len(train), 3), '  test:', round(dl/len(test), 3))
        count = 0
        for j in range(len(test)):
            pred = sess.run(output, feed_dict={inputs: test[j][0]})
            pred_len = len(pred)
            max_pred = list(pred[pred_len-1]).index(max(list(pred[pred_len-1])))
            max_test = list(test[j][1][0]).index(max(list(test[j][1][0])))
            if max_pred == max_test:
                count += 1
        print('test accuracy: ', round(count/len(test), 3))
        saver.save(sess, "params/lstm_amplitude_lwy.ckpt")


    t0 = time.time()
    train_epoch(30)
    t1 = time.time()
    print(" %f min" % round((t1 - t0)/60, 2))
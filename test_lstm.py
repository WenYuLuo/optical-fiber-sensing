from lstm_ofa_classify import *

# # 线路1
# dict = {0: '', 1: '', 2: '', 3: ''}
# # dict[0] = "/media/fish/Elements/Project/光纤传感/光纤音频/布放光缆"
# # dict[1] = "/media/fish/Elements/Project/光纤传感/光纤音频/机械施工"
# dict[2] = "/media/fish/Elements/Project/光纤传感/2018.10.9音频样本_去噪/线路1/人工井内施工"
# dict[3] = "/media/fish/Elements/Project/光纤传感/2018.10.9音频样本_去噪/线路1/下雨告警"
# print('线路1 testing ...')

# 线路2
dict = {0: '', 1: '', 2: '', 3: ''}
# dict[0] = "/media/fish/Elements/Project/光纤传感/2018.10.9音频样本_去噪/线路2/放缆"
# # dict[1] = "/media/fish/Elements/Project/光纤传感/光纤音频/机械施工"
# dict[2] = "/media/fish/Elements/Project/光纤传感/2018.10.9音频样本_去噪/线路2/人工井内施工"
dict[3] = "/media/fish/Elements/Project/光纤传感/2018.10.9音频样本_去噪/线路2/下雨"
print('线路2 testing ...')

test = []
slience_label = 4

for key in dict:
    print(dict[key])
    if dict[key] == '':
        continue
    count = 0
    slience_count =0
    wav_files = read_wav.list_wav_files(dict[key])

    for pathname in wav_files:
        wave_data, frameRate = read_wav.read_wav_file(pathname)

        wave_data = highpass(wave_data, frameRate, fl=1000)  # 高通滤波(若为多通道仅使用第一通道数据)

        if key == 3:
            active_pos_list = [[0, len(wave_data)]]
        else:
            active_pos_list = vad_process(wave_data)

        wave_data = wave_data.T

        wava_mean = float(np.sqrt(np.sum(wave_data ** 2)) / len(wave_data))

        ref_value = (2 ** 15 - 1) / wava_mean
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
            sample = data_enframe(clip_data, key)
            count += sample[0].shape[1]
            test.append(sample)

            if i == 0 and pos[0] != 0:
                # 起始静音段
                slience_start = 0
                slience_end = pos[0]
                clip_data = wave_data[slience_start:slience_end]
                sample = data_enframe(clip_data, slience_label)
                slience_count += sample[0].shape[1]
                test.append(sample)

            if i + 1 < len(active_pos_list):
                # 静音段
                slience_start = pos[1]
                slience_end = active_pos_list[i + 1][0]
                clip_data = wave_data[slience_start:slience_end]
                sample = data_enframe(clip_data, slience_label)
                slience_count += sample[0].shape[1]
                test.append(sample)
    print(count)

print('num of test sequences:%s' %len(test))    # 96
print('shape of inputs:', test[0][0].shape)     # (1,1722,512)
print('shape of labels:', test[0][1].shape)     # (1,4)

D_input = 512
D_label = 5
learning_rate = 7e-5
num_units = 1024

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

saver = tf.train.Saver()

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

saver.restore(sess, "params/lstm_amplitude_lwy.ckpt")

count = 0
for j in range(len(test)):
    pred = sess.run(output, feed_dict={inputs: test[j][0]})
    pred_len = len(pred)
    max_pred = list(pred[pred_len - 1]).index(max(list(pred[pred_len - 1])))
    pred_list = []
    for pre_x in pred:
        max_pre_x = list(pre_x).index(max(list(pre_x)))
        pred_list.append(max_pre_x)
    max_test = list(test[j][1][0]).index(max(list(test[j][1][0])))
    print(pred_list)
    print('ground truth:', max_test)
    if max_pred == max_test:
        count += 1
print('correct classified audio: %d, total: %d, test accuracy: %f' % (count, len(test), round(count / len(test), 3)))

from lstm_ofa_classify import *

# 线路1
dict = {0: '', 1: '', 2: '', 3: ''}
# dict[0] = "/media/fish/Elements/Project/光纤传感/光纤音频/布放光缆"
# dict[1] = "/media/fish/Elements/Project/光纤传感/光纤音频/机械施工"
dict[2] = "/media/fish/Elements/Project/光纤传感/2018.10.9音频样本/线路1/人工井内施工"
dict[3] = "/media/fish/Elements/Project/光纤传感/2018.10.9音频样本/线路1/下雨告警"
print('线路1 testing ...')

# # 线路2
# dict = {0: '', 1: '', 2: '', 3: ''}
# dict[0] = "/media/fish/Elements/Project/光纤传感/2018.10.9音频样本/线路2/放缆"
# # dict[1] = "/media/fish/Elements/Project/光纤传感/光纤音频/机械施工"
# dict[2] = "/media/fish/Elements/Project/光纤传感/2018.10.9音频样本/线路2/人工井内施工"
# dict[3] = "/media/fish/Elements/Project/光纤传感/2018.10.9音频样本/线路2/下雨"
# print('线路2 testing ...')

test = []

for key in dict:
    print(dict[key])
    if dict[key] == '':
        continue
    count = 0
    wav_files = read_wav.list_wav_files(dict[key])

    for pathname in wav_files:
        wave_data, frameRate = read_wav.read_wav_file(pathname)

        wave_data = highpass(wave_data, frameRate, fl=1000)  # 高通滤波

        wave_data = wave_data.T

        wava_mean = float(np.sqrt(np.sum(wave_data ** 2)) / len(wave_data))

        ref_value = (2 ** 15 - 1) / wava_mean
        wave_data = wave_data / ref_value  # wave幅值归一化

        nw = 512
        inc = 256
        win_fun = np.hamming(nw)
        frames = enframe(wave_data, nw, inc, win_fun)  # (1722，512) 1722帧，每帧长度512，每帧间隔长度256

        frames = np.fft.fft(frames)
        frames = np.sqrt(frames.real**2 + frames.imag ** 2)
        frames = np.expand_dims(np.expand_dims(frames, axis=0), axis=0)
        frames = list(frames)

        label = [0, 0, 0, 0]
        label[key] = 1
        label = np.array([[label]])
        label = list(label)
        sample = frames + label

        count += 1

        test.append(sample)

print('num of test sequences:%s' %len(test))    # 96
print('shape of inputs:', test[0][0].shape)     # (1,1722,512)
print('shape of labels:', test[0][1].shape)     # (1,4)

D_input = 512
D_label = 4
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
    max_test = list(test[j][1][0]).index(max(list(test[j][1][0])))
    if max_pred == max_test:
        count += 1
print('correct classified audio: %d, total: %d, test accuracy: %f' % (count, len(test), round(count / len(test), 3)))

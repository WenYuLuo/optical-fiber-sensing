import time
import shutil
import common
import hmm_optical_sensing
import pyodbc
import json
import os
import sys


class MsSQL:
    def __init__(self, host, user, pwd, db):
        self.host = host
        self.user = user
        self.pwd = pwd
        self.db = db
        self.connection = 'DRIVER={SQL Server};SERVER=%s;DATABASE=%s;UID=%s;PWD=%s' \
                          % (host, db, user, pwd)

    def __GetConnect(self):
        if not self.db:
            raise(NameError, "没有设置数据库信息")
        self.conn = pyodbc.connect(self.connection)
        cur = self.conn.cursor()
        if not cur:
            raise(NameError, "连接数据库失败")
        else:
            return cur

    def ExecQuery(self, sql):
        cur = self.__GetConnect()
        cur.execute(sql)
        resList = cur.fetchall()
        #查询完毕后必须关闭连接
        self.conn.close()
        return resList

    def ExecNonQuery(self, sql):
        cur = self.__GetConnect()
        cur.execute(sql)
        self.conn.commit()
        self.conn.close()

    def InsertAlarm(self, table_name, date, alarmtype, audio_save_path):
        # 避免数据库键值重复，导致无法插入
        date = date[:8] + ' ' + date[8:10] + ':' + date[10:12] + ':' + date[12:14] + '.003'
        sql = "INSERT INTO %s (LEVEL, TIME, ALARMTYPE, AUDIOPATH) VALUES " \
              "(1, " \
              "CAST('%s' AS DATETIME)," \
              " '%s', '%s')" \
              % (table_name, date, alarmtype, audio_save_path)
        self.ExecNonQuery(sql)


def Run():
    with open('sys_config.json', encoding='UTF-8') as file:
        data = json.load(file)
    src_path = data['src_path']
    dst_path = data['dst_path']
    hmm_param = data['hmm_param']
    hmm_model_path = data['hmm_model']
    seg = hmm_param['seg']  # hmm分段
    nw = hmm_param['nw']  # 帧长
    n_mfcc = hmm_param['n_mfcc']  # mfcc维数
    # hmm 加载初始化
    print("loading hmm models")
    hmms = hmm_optical_sensing.hmms()
    model_list = common.find_ext_files(hmm_model_path, ext='.npy')
    for model_file in model_list:
        hmms.load_one_model(model_file)  # 加载模型
    print("hmm models is loaded.")
    # 根据类别名创建文件夹
    print("building folder for alarm audio")
    for name in hmms.model_name:
        dir = os.path.join(dst_path, name)
        common.mkdir(dir)
    print("folder build.")

    # SQL server 初始化
    sql_param = data['sql_param']
    isconnect = int(sql_param['isconnect'])
    if isconnect == 1:
        host = sql_param['host']
        user = sql_param['user']
        pwd = sql_param['pwd']
        db = sql_param['db']
        table = sql_param['table']
        ms = MsSQL(host=host, user=user, pwd=pwd, db=db)
    elif isconnect == 0:
        ms = None
        table = 'empty'
    else:
        raise ValueError('isconnect value error! it should be 0 or 1!')

    # ms.InsertAlarm(' ', ' ', ' ', ' ')
    print("start detecting")
    count = 0
    while True:
        count += 1
        if count % 5 == 0:
            print('\b\b\b\b', end='')
            print('    ', end='')
            print('\b\b\b\b', end='')
        else:
            print('.', end='')
        sys.stdout.flush()
        wav_list = common.find_ext_files(src_path, ext='.wav')
        time.sleep(1)
        for wav in wav_list:
            filename = wav.split('\\')[-1]
            try:
                predict_result = hmms.predict_wav(wav, seg=seg, nw=nw, n_mfcc=n_mfcc)
            except Exception as e:
                print('\rfile %s detection error %s' % (filename, e))
                continue
            result = hmms.model_name[predict_result]
            audio_save_path = os.path.join(dst_path, result, filename)
            shutil.move(wav, audio_save_path)
            # print("Event Occur: %s---%s" % (filename, result))
            print(("\rEvent Occur: %s---%s" % (filename, result)))
            # 添加到数据库
            if ms is not None:
                date = filename.split('.')[0]
                ms.InsertAlarm(table, date=date, alarmtype=result, audio_save_path=audio_save_path)
        # 确保当前list的wav文件已被处理，不被重复处理。
        for wav in wav_list:
            if os.path.exists(wav):
                os.remove(wav)


if __name__ == '__main__':
    while True:
        try:
            Run()
        except Exception as e:
            print(e)
            print('restart!')

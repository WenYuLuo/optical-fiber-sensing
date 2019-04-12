# encoding=utf-8
from logging.handlers import TimedRotatingFileHandler
import win32timezone
import win32serviceutil
import win32service
import win32event
import servicemanager
import os
import sys
import logging
import inspect
import time
import shutil
from myModule import common
from myModule import hmm_optical_sensing
import pyodbc
import json


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

    def InsertAlarm(self, table_name, date, loc, alarmtype, audio_save_path):
        sql = "INSERT INTO %s (LEVEL, TIME, ALARMTYPE, TJLOC, AUDIOPATH) VALUES " \
              "(1, CAST(STUFF(STUFF(STUFF('%s',9,0,' ' ),12,0,':'),15,0,':') AS DateTime), '%s', '%s', '%s')" \
              % (table_name, date, alarmtype, loc, audio_save_path)
        self.ExecNonQuery(sql)


class OpticalSensingService(win32serviceutil.ServiceFramework):
    _svc_name_ = "OpticalSensingService"                    #服务名
    _svc_display_name_ = "OpticalSensing"                 #job在windows services上显示的名字
    _svc_description_ = "Optical Sensing Recognition"        #job的描述

    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
        self.logger = self._getLogger()
        self.T = time.time()
        self.run = True

    def _getLogger(self):
        '''日志记录'''
        logger = logging.getLogger('[OpticalSensingService]')
        this_file = inspect.getfile(inspect.currentframe())
        dirpath = os.path.abspath(os.path.dirname(this_file))
        if os.path.isdir('%s\\log' % dirpath):  # 创建log文件夹
            pass
        else:
            os.mkdir('%s\\log' % dirpath)
        dir = '%s\\log' % dirpath
        print(dir)

        handler = TimedRotatingFileHandler(os.path.join(dir, "OpticalSensing.log"), when="midnight", interval=1,
                                           backupCount=20)
        formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        return logger

    def SvcDoRun(self):
        self.logger.info("service is run....")
        try:
            data = {}
            with open('C:\\test.json', encoding='UTF-8') as file:
                data = json.load(file)
            src_path = data['src_path']
            dst_path = data['dst_path']
            hmm_param = data['hmm_param']
            hmm_model = data['hmm_model']
            seg = hmm_param['seg']  # hmm分段
            nw = hmm_param['nw']  # 帧长
            n_mfcc = hmm_param['n_mfcc']  # mfcc维数
            # hmm 加载初始化
            self.logger.info("loading hmm models")
            hmms = hmm_optical_sensing.hmms()
            hmms.load_model(hmm_model)  # 加载模型
            self.logger.info("hmm models is loaded.")
            # 根据类别名创建文件夹
            self.logger.info("building folder for alarm audio")
            for name in hmms.model_name:
                dir = os.path.join(dst_path, name)
                common.mkdir(dir)
            self.logger.info("folder build.")

            # SQL server 初始化
            sql_param = data['sql_param']
            host = sql_param['host']
            user = sql_param['user']
            pwd = sql_param['pwd']
            db = sql_param['db']
            table = sql_param['table']
            ms = MsSQL(host=host, user=user, pwd=pwd, db=db)
            self.logger.info("start detecting")
            while self.run:
                wav_list = common.find_ext_files(src_path, ext='.wav')
                time.sleep(1)
                for wav in wav_list:
                    filename = wav.split('\\')[-1]
                    predict_result = hmms.predict_wav(wav, seg=seg, nw=nw, n_mfcc=n_mfcc)
                    result = hmms.model_name[predict_result]
                    audio_save_path = os.path.join(dst_path, result, filename)
                    shutil.move(wav, audio_save_path)
                    self.logger.info("Event Occur: %s---%s" % (filename, result))
                    #添加到数据库
                    ms.InsertAlarm(table, date=filename.split('.')[0], loc='默认', alarmtype=result, audio_save_path=audio_save_path)
                # 确保当前list的wav文件已被处理，不被重复处理。
                for wav in wav_list:
                    if os.path.exists(wav):
                        os.remove(wav)

        except Exception as e:
            self.logger.info(e)
            time.sleep(60)

    def SvcStop(self):
        self.logger.info("service is stop....")
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.hWaitStop)
        self.run = False


if __name__ == '__main__':
    if len(sys.argv) == 1:
        servicemanager.Initialize()
        servicemanager.PrepareToHostSingle(OpticalSensingService)
        servicemanager.StartServiceCtrlDispatcher()
    else:
        win32serviceutil.HandleCommandLine(OpticalSensingService)
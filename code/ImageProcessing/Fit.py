import cv2
import os
import numpy as np
import tensorflow as tf
from scipy import *
from scipy.linalg import norm, pinv
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance
import time
from functools import wraps
import shutil
from sklearn.preprocessing import Imputer
from ImageData import ImageDataProcess,func_timer

class xgboostmodel(object):
    def __init__(self,FilePath):
        self.path = FilePath

    def ReadData(self):
        self.data = self.loadDataset(self.path)
        self.TrainData,self.xtrain, self.ytrain = self.featureSet(self.data)
        # self.TestData = self.loadTestData(self.data)

    def loadDataset(self, FilePath):
        FileName = FilePath+'/data.txt'
        print('reading ...'+FileName)
        df = pd.read_csv(FilePath+'/data.txt', sep='\t', header=None)
        df.columns = ['X','Y','x','y','A','B','G']
        return df

    @func_timer
    def featureSet(self, data):  # 5 ==> 2
        data_num = len(data)
        XList = []
        print('featureSet')
        def makelist(x):
            tmp_list = []
            tmp_list.append(x['X'])
            tmp_list.append(x['Y'])
            tmp_list.append(x['A'])
            tmp_list.append(x['B'])
            tmp_list.append(x['G'])
            XList.append(tmp_list)

        data.apply(makelist, axis = 1)
        xList = data.x.values
        yList = data.y.values
        return XList, xList, yList

    @func_timer
    def loadTestData(self,data):
        data_num = len(data)
        XList = []
        def makelist(x):
            tmp_list = []
            tmp_list.append(x['X'])
            tmp_list.append(x['Y'])
            tmp_list.append(x['A'])
            tmp_list.append(x['B'])
            tmp_list.append(x['G'])
            XList.append(tmp_list)
        data.apply(makelist, axis=1)
        return XList

    @func_timer
    def train2D(self):
        # XGBoost训练过程
        print('fitting model x')
        # self.model = xgb.XGBRegressor(max_depth=6, learning_rate=0.1, n_estimators=1600,  objective='reg:gamma')
        self.model = xgb.XGBRegressor(colsample_bytree = 0.4603, gamma = 0.0468,
        learning_rate = 0.05, max_depth = 4,
        min_child_weight = 2, n_estimators = 2200,
        reg_alpha = 0.4640, reg_lambda = 0.8571,
        subsample = 0.5213, silent = 0,
        random_state = 7, nthread = -1)
        #self.a = [1200,1100,1000]
        self.model.fit(float64(self.TrainData),float64(self.xtrain))
        #self.model.fit([[1200,3300,103,103,103],[1100,3300,103,103,103],[1000,3300,103,103,103]],self.xtrain)
        #self.model.fit(self.TrainData, self.xtrain)
        print('fitting model y')
        self.model2 = xgb.XGBRegressor(max_depth=6, learning_rate=0.1, n_estimators=1600,  objective='reg:gamma', silent=0)
        self.model2.fit(float64(self.TrainData),float64(self.xtrain))

    def PredictAll(self):
        # 对测试集进行预测
        ansx = self.model.predict(xgb.DMatrix(self.TrainData))
        ansy = self.model2.predict(xgb.DMatrix(self.TrainData))
        ans_len = len(ansx)
        data_arr = []
        for row in range(0, ans_len):
            tmp=[]
            tmp.extend(self.TrainData[row])
            tmp.append(ansx[row])
            tmp.append(ansy[row])
            data_arr.append(tmp)
        np_data = np.array(data_arr)

        # 写入文件
        pd_data = pd.DataFrame(np_data, columns=['X', 'Y', 'alpha', 'beta', 'gama', 'predict_x', 'predict_y'])
        # print(pd_data)
        pd_data.to_csv(self.path+'/'+'predictall.csv', index=None)

        # 显示重要特征
        #plot_importance(self.model)
        #plt.show()

    def Predict(self, Input): ### 非常坑
        ansx = self.model.predict(xgb.DMatrix(Input))
        ansy = self.model2.predict(xgb.DMatrix(Input))
        ans_len = len(ansx)
        result = []
        for row in range(0, ans_len):
            result.append([ansx[row],ansy[row]])
        return result

    def ModelSave(self):

        self.model.get_booster().save_model(self.path+'/'+'model1.model')
        self.model2.get_booster().save_model(self.path + '/' + 'model2.model')

    def ModelReload(self):
        self.model = xgb.Booster(model_file=self.path+'/'+'model1.model')
        self.model2 = xgb.Booster(model_file=self.path+'/'+'model2.model')


# 召唤企鹅
# a = DongGua('image_253_3_53.bmp')

# 测试数据转换
# points_result = DongGuaProcessing.ImgToArray('image_253_3_53.bmp')

# 数据初处理
# DongGuaProcessing.EatAllDongGua('E://imgimporigin')  # 拿到两个文件
# DongGuaProcessing.EatBadDongGua('E://imgimporigin', 'E://imgimgbad') # 挑出坏文件

# xgboost拟合
xmodel = xgboostmodel('E:/imgimporigin')

xmodel.ReadData()

xmodel.train2D() # 5映2训练
# xmodel.ModelSave()

xmodel.ModelReload()


xmodel.PredictAll() # 预测所有的点 用于计算精度 放在文件里
target = int64([[500,3300,103,103,103],
          [600,3300,103,103,103],
          [700,3300,103,103,103],
          [800,3300,103,103,103],
          [900,3300,103,103,103],
          [0, 2700, 103, 103, 103]])
print(xmodel.Predict(target))

ImageDataProcess.CircleDongGua('E:/imgimporigin/image_103_103_103.bmp',xmodel)
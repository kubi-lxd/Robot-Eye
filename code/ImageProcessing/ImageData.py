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
from ImageProcess import MyException


def func_timer(function):
    '''
    用装饰器实现函数计时
    :param function: 需要计时的函数
    :return: None
    '''
    @wraps(function)
    def function_timer(*args, **kwargs):
        print('[Function: {name} start...]'.format(name=function.__name__))
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print('[Function: {name} finished, spent time: {time:.2f}s]'.format(name=function.__name__, time=t1 - t0))
        return result
    return function_timer


class ImageDataProcess(object):
    def calculateH(self, ImgName):
        head, alpha, beta, gamaend = ImgName.split('_')
        fig = cv2.imread(ImgName)
        # cv2.imshow('original fig', fig)
        # 原图中的4个点
        src_point = np.float32([[50, 100], [150, 100], [50, 200], [150, 200]])
        dst_point = np.float32([[0, 0], [1000, 0], [0, 1000], [1000, 1000]])
        # 至少要4个点，一一对应，找到映射矩阵h
        h, s = cv2.findHomography(src_point, dst_point, cv2.RANSAC, 10)
        # test
        '''
        h = np.mat(h)
        o = np.mat([[150], [200], [1]])
        res = h * o
        res = np.asarray(res)
        res = res.astype(np.int32)'''
        return h

    @classmethod
    def imgtoarray(cls, ImgName):
        data = ImageDataProcess(ImgName)
        ArrayResult = []
        for ColorsList in data.outpoints:
            for Point in ColorsList:
                ArrayResult.append(Point)
        return ArrayResult

    @classmethod
    def datatrans(cls, Type, *args):
        assert Type in ['RBF', 'MultiColumn'], "非法类型"
        if Type == 'RBF':
            DataList= []
            for arg in args:
                arg = np.array(arg)
                arg = arg[:, np.newaxis]
                DataList.append(arg)
            RbfResult = np.dstack(DataList)
            RbfResult = RbfResult.swapaxes(1,0)
            return RbfResult
        if Type == 'MultiColumn':
            PointsResults = args[0]
            X = []
            Y = []
            A = []
            B = []
            G = []
            xlist = []
            ylist = []
            for point in PointsResults:
                X.append(point[0])
                Y.append(point[1])
                xlist.append(point[2])
                ylist.append(point[3])
                A.append(point[4])
                B.append(point[5])
                G.append(point[6])
            return cls.DataTrans('RBF',X,Y,A,B,G),xlist,ylist

    @classmethod
    @func_timer
    def solveallfigures(cls, PathName):
        TotalResult = []
        with open(PathName + '/'+'data.txt','w') as DataFile:
            with open(PathName + '/'+'error.txt','w') as ErrorFile:  ## !!!
                for filename in os.listdir(PathName):
                    if filename[-3:] == 'bmp':
                        print('reading:'+filename+'...')
                        ImgName = PathName + '/'+filename
                        try:
                            Array = cls.ImgToArray(ImgName)
                            for data in Array:
                                data_out = str(data).strip('[').strip(']').replace(',', '\t')+'\n'
                                if len(data_out.split('\t'))==7:
                                    DataFile.write(data_out)
                                else:
                                    print('errorfile:  ' + filename + ' ' + str(data))
                                    ErrorFile.write(filename + '  ' + str('DataError') + '\n')
                                    break
                            TotalResult.extend(Array)
                        except Exception as e:
                            print('error:'+filename,e)
                            ErrorFile.write(filename+'  '+str(e)+'\n')
        return TotalResult

    @classmethod
    def finderrorfiles(cls,PathName,NewPath):  # cp bad files to one folder
        with open(PathName + '/' + 'error.txt', 'r', encoding='UTF-8') as ErrorFile:
            for i, line in enumerate(ErrorFile):
                ImgName, ErrorName=line.strip('  ').strip('\n').split('.bmp')
                shutil.copyfile(PathName+'/'+ImgName+'.bmp', NewPath+'/'+ImgName+'.bmp')  # 复制

    @classmethod
    def Circlefigure(cls,ImgName,modelclass):
        head, alpha, beta, gamaend = ImgName.split('_')
        gama, end = gamaend.split('.')
        # 转整数
        alpha = int(alpha)
        beta = int(beta)
        gama = int(gama)
        # 3m circle
        r=3000
        x0=0
        y0=0
        angle = [x*1 for x in range(0,181)]
        CirclePointList = []
        for a in angle:
            x1 = x0 + r * cos(a * pi / 180)
            y1 = y0 + r * sin(a * pi / 180)
            CirclePointList.append([x1,y1,alpha,beta,gama])
        FigPointList = modelclass.Predict(float64(CirclePointList))
        RealFigPoints = []
        for FigPoint in FigPointList:
            FigPoint[0] = round(FigPoint[0])
            FigPoint[1] = round(FigPoint[1])
            if FigPoint[0]<=0 or FigPoint[0]>=1920: continue
            if FigPoint[1]<=0 or FigPoint[0]>=1080: continue
            RealFigPoints.append([FigPoint[0],FigPoint[1]])

        # 画近似曲线
        img = cv2.imread(ImgName)
        for p in RealFigPoints:
            cv2.circle(img, (p[0], p[1]), 15, (0, 0, 255), 2)
        '''
        cv2.imshow('curve',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        return img
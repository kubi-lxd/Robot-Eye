import shutil
import time
from functools import wraps

import cv2
import numpy as np
import os
from scipy import *
from ImageProcess import ImageProcess
from Exception import MyException
OriginalImgPath = '../../figures/figures/'
BadImgPath = '../../figures/figures/bad files/'
# img = cv2.copyMakeBorder(img, 200, 200, 200, 200, cv2.BORDER_CONSTANT, 0)

class HError(MyException):
    def __init__(self):
        super(HError, self).__init__(code=1003, message='not enough points to calculate H matrix', args=('HError,',))


def func_timer(function):
    """
    用装饰器实现函数计时
    :param function: 需要计时的函数
    :return: None
    """
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
    @classmethod
    def imgtoarray(cls, imgname):
        data = ImageProcess(imgname)
        arrayresult = []
        for ColorsList in data.outpoints:
            for Point in ColorsList:
                arrayresult.append(Point)
        return arrayresult

    @classmethod
    def calculateHmatrix(cls, imgname):
        arrayresult = cls.imgtoarray(imgname)
        src_point = []
        dst_point = []
        for point in arrayresult:
            realworldpos = [point[0], point[1]]
            pixelworldpos = [point[2], point[3]]
            src_point.append(realworldpos)
            dst_point.append(pixelworldpos)
        if len(src_point) <= 3 or len(src_point) != len(dst_point):
            raise HError
        print('src:')
        print(src_point)
        print('dst:')
        print(dst_point)
        h, s = cv2.findHomography(np.float32(src_point), np.float32(dst_point), cv2.RANSAC, 10)
        '''
                h = np.mat(h)
                o = np.mat([[150], [200], [1]])
                res = h * o
                res = np.asarray(res)
                res = res.astype(np.int32)
                '''
        return h

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
            return cls.datatrans('RBF',X,Y,A,B,G), xlist, ylist

    @classmethod
    @func_timer
    def solveallfigures(cls, filepath):
        totalresult = []
        with open(filepath + 'data.txt', 'w') as DataFile:
            with open(filepath + 'error.txt', 'w') as ErrorFile:
                for filename in os.listdir(filepath):
                    if filename[-3:] == 'bmp':
                        print('reading:'+filename+'...')
                        imgname = filepath + '/'+filename
                        try:
                            Array = cls.imgtoarray(imgname)
                            for data in Array:
                                data_out = str(data).strip('[').strip(']').replace(',', '\t')+'\n'
                                if len(data_out.split('\t')) == 7:
                                    DataFile.write(data_out)
                                else:
                                    print('errorfile:  ' + filename + ' ' + str(data))
                                    ErrorFile.write(filename + '  ' + str('DataError') + '\n')
                                    break
                            totalresult.extend(Array)
                        except Exception as error:
                            print('error:'+filename, error)
                            ErrorFile.write(filename+'  '+str(e)+'\n')
        return totalresult

    @classmethod
    def finderrorfiles(cls, filepath, newpath):  # copy bad files to one folder
        print('finderrorfiles')
        fignum =0
        with open(filepath + '/' + 'error.txt', 'r', encoding='UTF-8') as ErrorFile:
            for i, line in enumerate(ErrorFile):
                fignum += 1
                imgname, errorname=line.strip('  ').strip('\n').split('.bmp')
                shutil.copyfile(filepath+'/'+imgname+'.bmp', newpath+'/'+imgname+'.bmp')
        print('find' + str(fignum) + 'files')

    @classmethod
    def Circlefigure(cls, imgname, modelclass):
        head, alpha, beta, gamaend = imgname.split('_')
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
            CirclePointList.append([x1, y1, alpha, beta, gama])
        FigPointList = modelclass.Predict(float64(CirclePointList))
        RealFigPoints = []
        for FigPoint in FigPointList:
            FigPoint[0] = round(FigPoint[0])
            FigPoint[1] = round(FigPoint[1])
            if FigPoint[0] <= 0 or FigPoint[0] >= 1920:
                continue
            if FigPoint[1] <= 0 or FigPoint[0] >= 1080:
                continue
            RealFigPoints.append([FigPoint[0], FigPoint[1]])

        # 画近似曲线
        img = cv2.imread(imgname)
        for p in RealFigPoints:
            cv2.circle(img, (p[0], p[1]), 15, (0, 0, 255), 2)

        '''
        cv2.imshow('curve',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        return img


if __name__ == '__main__':
    OriginalImgPath = '../../figures/examples/'
    BadImgPath = '../../figures/examples/bad files/'
    Data = ImageDataProcess.imgtoarray('image_253_3_53.bmp')
    print('ImgArray:')
    print(Data)
    H = ImageDataProcess.calculateHmatrix('image_253_3_53.bmp')
    print('Hmatrix:')
    print(H)
    point = Data[4]
    o = np.mat([[point[0]], [point[1]], [1]])
    print('real coordinate:')
    print([point[2], point[3]])
    res = H * o
    res = np.asarray(res)
    res = res / res[2]
    res = res.astype(np.int32)
    print('Hpredict:')
    print(res)
    ImageDataProcess.solveallfigures(OriginalImgPath)
    ImageDataProcess.finderrorfiles(OriginalImgPath, BadImgPath)
    # Circlefigure()
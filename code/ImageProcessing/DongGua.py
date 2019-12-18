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


class DongGua():
    def __init__(self, name):
        self.name = name
        self.img0 = cv2.imread(name, 0)
        self.img1 = cv2.imread(name)
        self.GetABC()
        self.mark = self.ReadMark()
        if (len(self.mark) == 2):
            self.flag = 'middle'
            self.midmark = []
        elif (len(self.mark) == 1 and self.mark[0][0] == -3500):
            self.flag = 'left'
        elif (len(self.mark) == 1 and self.mark[0][0] == 1200):
            self.flag = 'right'
        self.allpoint = []
        self.colorpoint = dict()
        self.outpoints = []
        self.lines = []
        self.getxy()
        self.getcolor()
        self.getlines()
        self.getrows()
        self.getrealposition()


    def GetABC(self):
        head, alpha, beta, gamaend = self.name.split('_')
        gama, end = gamaend.split('.')
        # 转整数
        self.alpha = int(alpha)
        self.beta = int(beta)
        self.gama = int(gama)

    def ReadMark(self):
        """
        :param figname: 图片名称 典型值'image_178_3_153.bmp'
        :return: 数组list 典型值[[-600, 1800, 1141.3, 298.25], [-600, 1500, 1085.8, 430.25]]
        数组的长度表示标注点的数量 每个标注点包含四个参数 X Y x y 大写为真实坐标 小写为图像坐标
        xy为小数是因为标注时图片太大经过缩放 产生了小数坐标标注 参考小数坐标寻找最临近亮点即可
        """

        with open('batch mark result.txt', 'r', encoding='UTF-8') as f:
            result = []
            lenth = 0
            for i, line in enumerate(f):
                X, Y, x, y,alphahere,betahere,gamahere,num = line.split('\t')
                X = int(X)
                Y = int(Y)
                x= float(x)
                y= float(y)
                alphahere = int(alphahere)
                betahere = int(betahere)
                gamahere = int(gamahere)
                num = int(num)
                if alphahere== self.alpha and betahere==self.beta and gamahere==self.gama:
                    result.append([X,Y,x,y])
                    lenth = num
            if not len(result)==lenth:
                print('error NOOOOOOO!')
                return None
            else:
                return result

    def change(self, x, y, xsize, ysize):
        if x <= 0:
            x = 1
        if y <= 0:
            y = 1
        if x >= xsize:
            x = xsize
        if y >= ysize:
            y = ysize
        return x, y

    def SeeSee(self):
        """
        输入图片名称 显示检查标注点 会打开一张图片
        :param figname:图片名称
        :return:无
        """
        fig = self.img1
        xsize = fig.shape[1]
        ysize = fig.shape[0]
        # 拿到标注点信息
        points = self.ReadMark()
        font = cv2.FONT_HERSHEY_SIMPLEX
        # 标点 画圈 写字
        if len(points):
            for point in points:
                p = self.change(int(point[2]), int(point[3]),xsize,ysize)
                cv2.circle(fig, (p[0],p[1]), 15, (0, 0, 255), 2)
                p = self.change(int(point[2]), int(point[3])-10,xsize,ysize)
                cv2.putText(fig, str(int(point[0]))+','+str(int(point[1])),
                           (p[0],p[1]), font, 1,(255, 255, 255), 1)
        cv2.namedWindow('see_image', cv2.WINDOW_AUTOSIZE)
        # 缩放 防止超出屏幕
        size = fig.shape
        fig = cv2.resize(fig, (int(size[1] * 0.5), int(size[0] * 0.5)), cv2.INTER_LINEAR)
        # 显示
        cv2.imshow('see_image', fig)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def getxy(self):
        self.allpoint = []
        blackimg = cv2.imread('black1.bmp', 0)
        ret, thresh1 = cv2.threshold(self.img0, 170, 255, cv2.THRESH_BINARY)
        kernel = np.ones((13, 8), np.uint8)
        dilation = cv2.dilate(thresh1, kernel, iterations=1)
        median = cv2.medianBlur(dilation, 7)
        contours, hierarchy = cv2.findContours(median.copy(), 1, 2)
        for c in contours:
            m = cv2.moments(c)
            cX = int(m["m10"] / m["m00"])
            cY = int(m["m01"] / m["m00"])
            cv2.circle(blackimg, (cX, cY), 3, 255, -1)
            self.allpoint.append([cX, cY])
        size = blackimg.shape
        blackimg = cv2.resize(blackimg, (int(size[1] * 0.5), int(size[0] * 0.5)), cv2.INTER_LINEAR)
        #cv2.imshow('sample', blackimg)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        print(len(self.allpoint))

    def getcolor(self):
        blackimg = cv2.imread('black1.bmp')
        blur = cv2.GaussianBlur(self.img1, (31, 31), 0)
        blue = []
        purple = []
        white = []
        red = []
        yellow = []
        green = []
        for c in self.allpoint:
            color = blur[c[1], c[0]]
            # color2 = blur[c[1], min(c[0] + 35, 1080)]
            color1 = {'b': int(color[0]), 'g': int(color[1]), 'r': int(color[2])}
            '''if (color1['b'] > 75) + (color1['g'] > 75) + (color1['r'] > 75) + \
                    (color2[0] > 40 or color2[1] > 40 or color2[2] > 40) >= 3:'''
            if (color1['b'] > 75) + (color1['g'] > 75) + (color1['r'] > 75) >= 2:
                if color1['b'] > color1['g'] + 30 and color1['b'] > color1['r'] + 30 \
                        and color1['g'] - color1['r'] > 10:
                    blue.append([c, 'blue'])
                    cv2.circle(blackimg, (c[0], c[1]), 3, (255, 100, 100), -1)
                elif color1['b'] > color1['g'] + 30 and color1['r'] - color1['g'] > 10:
                    purple.append([c, 'purple'])
                    cv2.circle(blackimg, (c[0], c[1]), 3, (255, 100, 255), -1)
                elif color1['b'] - color1['g'] > 15 and color1['b'] - color1['r'] > 15:
                    white.append([c, 'white'])
                    cv2.circle(blackimg, (c[0], c[1]), 3, (255, 255, 255), -1)
                elif color1['r'] - color1['b'] > 30 and color1['r'] - color1['g'] > 25:
                    red.append([c, 'red'])
                    cv2.circle(blackimg, (c[0], c[1]), 3, (100, 100, 255), -1)
                elif color1['r'] - color1['b'] > 25:
                    yellow.append([c, 'yellow'])
                    cv2.circle(blackimg, (c[0], c[1]), 3, (100, 255, 255), -1)
                elif color1['g'] - color1['r'] > 25 and color1['b'] - color1['r'] > 25:
                    green.append([c, 'green'])
                    cv2.circle(blackimg, (c[0], c[1]), 3, (100, 255, 100), -1)
        '''cv2.imshow('sample', blackimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()'''
        self.colorpoint['blue'] = blue
        self.colorpoint['purple'] = purple
        self.colorpoint['white'] = white
        self.colorpoint['red'] = red
        self.colorpoint['yellow'] = yellow
        self.colorpoint['green'] = green

    def getlines(self):
        self.outpoints = []
        blackimg2 = self.img1.copy()
        for key in self.colorpoint.keys():
            blackimg = cv2.imread('black1.bmp', 0)
            for p in self.colorpoint[key]:
                cv2.circle(blackimg, (p[0][0], p[0][1]), 3, 255, -1)
            length = max(20, int((len(self.allpoint)+100)/7))
            lines = cv2.HoughLines(blackimg, 1, np.pi / 180, length)
            if lines is not None:
                linea = self.findlines(lines)
                for line in linea:
                    rho, theta = line[0]
                    '''a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 2000 * (-b))
                    y1 = int(y0 + 2000 * (a))
                    x2 = int(x0 - 2000 * (-b))
                    y2 = int(y0 - 2000 * (a))
                    cv2.line(blackimg, (x1, y1), (x2, y2), 255, 2)
                    cv2.imshow('sample', blackimg)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()'''
                    pointcolor = []
                    blackimg = cv2.imread('black1.bmp', 0)
                    for p in self.allpoint:
                        if self.getdist(p, theta, rho) < 25:
                            p.append(key)
                            pointcolor.append(p)
                            cv2.circle(blackimg2, (p[0], p[1]), 3, 255, -1)
                    '''cv2.imshow('sample', blackimg2)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()'''
                    pointcolor.sort(key=lambda x: x[0], reverse=True)
                    self.outpoints.append(pointcolor)
        if self.flag == 'right' or self.flag == 'middle':
            self.outpoints.sort(key=lambda x: x[0][1])
        elif self.flag == 'left':
            self.outpoints.sort(key=lambda x: x[-1][1])
            flag = 0
            for i, line in enumerate(self.outpoints):
                if line[-1][2] == 'purple' and flag == 0:
                    self.outpoints[i+1].pop()
                    flag = 1
                elif line[-1][2] == 'green':
                    self.outpoints[i+1].pop()
                    break


    def getrows(self):
        if self.flag == 'middle':
            # gandianbiede
            p1 = [int(self.mark[0][2]), int(self.mark[0][3])]
            p2 = [int(self.mark[1][2]), int(self.mark[1][3])]
            aa = p1[1]-p2[1]
            bb = p2[0]-p1[0]
            cc = p1[0]*p2[1]-p1[1]*p2[0]
            '''x1 = int(p1[0] + 2000 * (bb))
            y1 = int(p1[1] + 2000 * (-aa))
            x2 = int(p1[0] - 2000 * (bb))
            y2 = int(p1[1] - 2000 * (-aa))
            cv2.line(self.img0, (x1, y1), (x2, y2), 255, 2)
            cv2.imshow('sample', self.img0)
            cv2.waitKey(0)
            cv2.destroyAllWindows()'''
            pindex = []
            for i, lines in enumerate(self.outpoints):
                for j, p in enumerate(lines):
                    dist = self.getdist2(p, aa, bb, cc)
                    if dist < 20:
                        pindex.append(j)
                        break
                if len(pindex) < i+1:
                    pindex.append(-1)
            blackimg = cv2.imread('black1.bmp', 0)
            for i, line in enumerate(self.outpoints):
                index = pindex[i]
                if index != -1:
                    cv2.circle(blackimg, (line[index][0], line[index][1]), 5, 255, -1)
            lines = cv2.HoughLines(blackimg, 1, np.pi / 180, 25)
            rho, theta = lines[0][0]
            '''a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 2000 * (-b))
            y1 = int(y0 + 2000 * (a))
            x2 = int(x0 - 2000 * (-b))
            y2 = int(y0 - 2000 * (a))
            cv2.line(self.img1, (x1, y1), (x2, y2), (255,255,255), 2)
            cv2.imshow('sample', self.img1)
            cv2.waitKey(0)
            cv2.destroyAllWindows()'''
            pindex = []
            for i, lines in enumerate(self.outpoints):
                ptm = []
                for j, p in enumerate(lines):
                    dist = self.getdist(p, theta, rho)
                    if dist < 80:
                        ptm.append([j, dist])
                    elif len(ptm) > 0:
                        pindex.append(self.findclose(ptm))
                        #cv2.circle(self.img1, (lines[pindex[i]][0], lines[pindex[i]][1]), 10, (255,255,255), 2)
                        break
                if len(pindex) < i+1:
                    if i != 0 and i != len(self.outpoints)-1:
                        pindex.append(pindex[i-1])
                    else:
                        pindex.append(-1)
            if pindex[0] == -1:
                self.outpoints[0] = [[-1, -1]]
            if pindex[-1] == -1:
                self.outpoints[-1] = [[-1, -1]]
            '''cv2.imshow('sample', self.img1)
            cv2.waitKey(0)
            cv2.destroyAllWindows()'''
            self.midmark = pindex
        else:
            for j in range(0, min(len(self.outpoints[0])-1, 4)):
                j = j if self.flag == 'right' else -1-j
                pointsr = [self.outpoints[0][j], self.outpoints[1][j], self.outpoints[2][j], self.outpoints[3][j]]
                blackimg = cv2.imread('black1.bmp', 0)
                for p in pointsr:
                    cv2.circle(blackimg, (p[0], p[1]), 3, 255, -1)
                lines = cv2.HoughLines(blackimg, 1, np.pi / 180, 12)
                if lines is not None:
                    rho, theta = lines[0][0]
                    for i, line in enumerate(self.outpoints):
                        if self.getdist(line[j], theta, rho) > 35:
                            if self.flag == 'right':
                                self.outpoints[i].insert(0, [-1, -1])
                            else:
                                self.outpoints[i].append([-1, -1])

    def getrealposition(self):
        img2 = cv2.imread(self.name)
        if self.flag == 'middle':
            pointm0 = self.mark[0]
            pointm1 = self.mark[1]
            pointm0[2] = int(pointm0[2])
            pointm0[3] = int(pointm0[3])
            pointm1[2] = int(pointm1[2])
            pointm1[3] = int(pointm1[3])
            points = []
            mark0 = [-1, -1]
            mark1 = [-1, -1]
            for i, line in enumerate(self.outpoints):
                index = self.midmark[i]
                if abs(line[index][0] - pointm0[2]) < 15 and abs(line[index][1] - pointm0[3]) < 15:
                    mark = [index, i]
                    #cv2.circle(self.img1, (line[index][0], line[index][1]), 10, (255, 255, 255), 2)
                elif abs(line[index][0] - pointm1[2]) < 15 and abs(line[index][1] - pointm1[3]) < 15:
                    mark1 = [index, i]
                    #cv2.circle(self.img1, (line[index][0], line[index][1]), 10, (255, 255, 255), 2)
            if mark0 == [-1, -1]:
                if mark1 == [-1, -1]:
                    return 0
                else:
                    mark = mark1
                    pointm = pointm1
            else:
                mark = mark0
                pointm = pointm0

            for i, line in enumerate(self.outpoints):
                linem = []
                j2 = self.midmark[i]
                for j, pointa in enumerate(line):
                    if pointa != [-1, -1]:
                        pm = [(j2 - j) * 100 + pointm[0], (mark[1] - i) * 300 + pointm[1], pointa[0], pointa[1]]
                        cv2.circle(img2, (pm[2], pm[3]), 10, (255,255,255), 2)
                        cv2.putText(img2, str(pm[0]), (pm[2] - 20, pm[3] - 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)
                        cv2.putText(img2, str(pm[1]), (pm[2] - 20, pm[3] - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)
                        linem.append(pm)
                if linem != []:
                    points.append(linem)
        else:
            pointm = self.mark[0]
            pointm[2] = int(pointm[2])
            pointm[3] = int(pointm[3])
            pointm[2], pointm[3] = self.change(pointm[2], pointm[3], 1920, 1080)
            points = []
            mark = [-1, -1]
            for i, line in enumerate(self.outpoints):
                index = 0 if self.flag == 'right' else -1
                print(line[index])
                if abs(line[index][0] - pointm[2]) < 15 and abs(line[index][1] - pointm[3]) < 15:
                    mark = [index, i]
                    break
            if mark == [-1, -1]:
                print(0)
                return 0
            for i, line in enumerate(self.outpoints):
                linem = []
                for j, pointa in enumerate(line):
                    j = j if self.flag == 'right' else -1-j
                    pointb = line[j]
                    if pointb != [-1, -1]:
                        pm = [(mark[0] - j) * 100 + pointm[0], (mark[1] - i) * 300 + pointm[1], pointb[0], pointb[1]]
                        cv2.circle(img2, (pm[2], pm[3]), 10, (255,255,255), 2)
                        cv2.putText(img2, str(pm[0]), (pm[2] - 20, pm[3] - 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)
                        cv2.putText(img2, str(pm[1]), (pm[2] - 20, pm[3] - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)
                        linem.append(pm)
                if linem != []:
                    points.append(linem)
        #cv2.imshow('sample', img2)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        for line in points:
            for p in line:
                p.append(self.alpha)
                p.append(self.beta)
                p.append(self.gama)
        self.outpoints = points
        #print(self.outpoints[0])
        #print(self.outpoints[-1])

    def findlines(self, lines):
        lineout = []
        if self.lines == []:
            self.lines.append(lines[0])
            lineout.append(lines[0])
        for linea in lines:
            difflag = 0
            rhoa = linea[0][0]
            for lineb in self.lines:
                rhob = lineb[0][0]
                if abs(rhoa - rhob) < 35:
                    difflag = 1
                    break
            if difflag == 0:
                self.lines.append(linea)
                lineout.append(linea)
        return lineout
    def getdist(self, point, theta, rho):
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        if b == 0:
            return abs(x0 - point[0])
        else:
            y = float((point[0] - x0) * (-a / b) + y0)
            return abs((y - point[1]) * b)
    def getdist2(self, point, a, b, c):
        return abs(a*point[0]+b*point[1]+c)/(a**2+b**2)**0.5
    def findclose(self, dist):
        aa = dist[0]
        for l in dist:
            if l[1] < aa[1]:
                aa = l
        return aa[0]


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
        print('[Function: {name} finished, spent time: {time:.2f}s]'.format(name = function.__name__,time = t1 - t0))
        return result
    return function_timer



class RBF:
    def __init__(self, indim, numCenters, outdim):
        self.indim = indim
        self.outdim = outdim
        self.numCenters=numCenters
        self.centers=[random.uniform(-1,1,indim)for i in range(numCenters)]
        self.beta=8
        self.W=random.random((self.numCenters,self.outdim))
        self.beta = 8
        self.W = random.random((self.numCenters, self.outdim))

    def _basisfunc(self, c, d):
        # assert len(d) == self.indim
        return exp(-self.beta * norm(c - d) ** 2)

    def _calcAct(self, X):
        # calculate activations of RBFs
        G = zeros((X.shape[0], self.numCenters), float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi, ci] = self._basisfunc(c, x)
        return G

    def train(self, X, Y):
        """ X: matrix of dimensions n x indim
            y: column vector of dimension n x 1 """

        # choose random center vectors from training set
        rnd_idx = random.permutation(X.shape[0])[:self.numCenters]
        self.centers = [X[i, :] for i in rnd_idx]

        print("center", self.centers)
        # calculate activations of RBFs
        G = self._calcAct(X)
        print(G)

        # calculate output weights (pseudoinverse)
        self.W = dot(pinv(G), Y)

    def test(self, X):
        """ X: matrix of dimensions n x indim """

        G = self._calcAct(X)
        Y = dot(G, self.W)
        return Y


class DongGuaProcessing(object):
    def calculateH(self, ImgName):
        head, alpha, beta, gamaend = ImgName.split('_')
        gama, end = gamaend.split('.')
        # 转整数
        alpha = int(alpha)
        beta = int(beta)
        gama = int(gama)
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
    def ImgToArray(cls, ImgName):
        data = DongGua(ImgName)
        ArrayResult = []
        for ColorsList in data.outpoints:
            for Point in ColorsList:
                ArrayResult.append(Point)
        return ArrayResult

    @classmethod
    def DataTrans(cls, Type, *args):
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
    def EatAllDongGua(cls, PathName):
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
                                    print('errorfile:  ' + filename + ' ' + str(data))  ## !!!
                                    ErrorFile.write(filename + '  ' + str('DataError') + '\n')
                                    break
                            TotalResult.extend(Array)
                        except Exception as e:
                            print('error:'+filename,e)
                            ErrorFile.write(filename+'  '+str(e)+'\n')
        return TotalResult

    @classmethod
    def EatBadDongGua(cls,PathName,NewPath):  # cp bad files to one folder
        with open(PathName + '/' + 'error.txt', 'r', encoding='UTF-8') as ErrorFile:
            for i, line in enumerate(ErrorFile):
                ImgName, ErrorName=line.strip('  ').strip('\n').split('.bmp')
                shutil.copyfile(PathName+'/'+ImgName+'.bmp', NewPath+'/'+ImgName+'.bmp')  # 复制

    @classmethod
    def CircleDongGua(cls,ImgName,modelclass):
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
        cv2.imshow('curve',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


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

DongGuaProcessing.CircleDongGua('E:/imgimporigin/image_103_103_103.bmp',xmodel)

'''
rbfx = RBF(2, 1000, 1)
rbfy = RBF(2, 1000, 1)

XXtest = np.array([[[-3500,3300]],[[-2300,2100]],[[-1100,900]]])

rbfx.train(XX, xlist)
rbfy.train(XX, ylist)


zx = rbfx.test(XXtest)
zy = rbfy.test(XXtest)




# plot original data
plt.figure(figsize=(12, 8))
plt.plot(zx, zy, 'k-')

# plot learned model
#plt.plot(X, zy, 'r-', linewidth=2)
plt.show()

# plot rbfs
plt.plot(rbf.centers, zeros(rbf.numCenters), 'gs')

for c in rbf.centers:
    # RF prediction lines
    cx = arange(c - 0.7, c + 0.7, 0.01)
    cy = [rbf._basisfunc(array([cx_]), array([c])) for cx_ in cx]
    plt.plot(cx, cy, '-', color='gray', linewidth=0.2)

plt.xlim(-1.2, 1.2)
plt.show()

# a.SeeSee()'''

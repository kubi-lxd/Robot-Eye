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
from Exception import MyException
MarkFilePath = '../../data/total/mark total.txt'
BlackImgPath = '../../examples/black1.bmp'
ResultImgPath = '../../examples/'


class MarkError(MyException):
    super().__init__(code=1001, message='MarkError', args=('MarkError',))


class ImageProcess():
    def __init__(self, name):
        self.name = name
        self.img0 = cv2.imread(name, 0)
        self.img1 = cv2.imread(name)

        self.midmark = None
        self.allpoint = []
        self.colorpoint = dict()
        self.outpoints = []
        self.lines = []
        self.flag = None
        self.midmark = []

        self.getabg(self.name)
        self.mark = self.readmark()
        self.gettype()
        self.getxy()

        self.getcolor()
        self.getlines()
        self.getrows()
        self.getrealposition()

    def getabg(self, name):
        head, alpha, beta, gamaend = name.split('_')
        gama, end = gamaend.split('.')
        # 转整数
        self.alpha = int(alpha)
        self.beta = int(beta)
        self.gama = int(gama)

    def readmark(self):
        with open(MarkFilePath, 'r', encoding='UTF-8') as f:
            result = []
            lenth = 0
            for i, line in enumerate(f):
                X, Y, x, y,alphahere,betahere,gamahere,num = line.split('\t')
                X = int(X)
                Y = int(Y)
                x = float(x)
                y = float(y)
                alphahere = int(alphahere)
                betahere = int(betahere)
                gamahere = int(gamahere)
                num = int(num)
                if alphahere== self.alpha and betahere==self.beta and gamahere==self.gama:
                    result.append([X,Y,x,y])
                    lenth = num
            if not len(result)==lenth:
                raise MarkError
            else:
                return result

    def gettype(self):
        if (len(self.mark) == 2):
            self.flag = 'middle'
            self.midmark = []
        elif (len(self.mark) == 1 and self.mark[0][0] == -3500):
            self.flag = 'left'
        elif (len(self.mark) == 1 and self.mark[0][0] == 1200):
            self.flag = 'right'

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

    def getxy(self,output=False,screen=False):
        blackimg = cv2.imread(BlackImgPath, 0)
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
        if screen:
            cv2.imshow('sample', blackimg)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print(len(self.allpoint))
        if output:
            cv2.imwrite(ResultImgPath+'getxy'+str(self.alpha)+'_'+str(self.beta)+'_'+str(self.gama)+'.bmp',blackimg)
        return blackimg

    def getcolor(self, output=False, screen=False):
        blackimg = cv2.imread(BlackImgPath, 0)
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
            # shining point
            if (color1['b'] > 75) + (color1['g'] > 75) + (color1['r'] > 75) >= 2:
                # blue
                if color1['b'] > color1['g'] + 30 and color1['b'] > color1['r'] + 30 \
                        and color1['g'] - color1['r'] > 10:
                    blue.append([c, 'blue'])
                    cv2.circle(blackimg, (c[0], c[1]), 3, (255, 100, 100), -1)
                # purple
                elif color1['b'] > color1['g'] + 30 and color1['r'] - color1['g'] > 10:
                    purple.append([c, 'purple'])
                    cv2.circle(blackimg, (c[0], c[1]), 3, (255, 100, 255), -1)
                # white
                elif color1['b'] - color1['g'] > 15 and color1['b'] - color1['r'] > 15:
                    white.append([c, 'white'])
                    cv2.circle(blackimg, (c[0], c[1]), 3, (255, 255, 255), -1)
                # red
                elif color1['r'] - color1['b'] > 30 and color1['r'] - color1['g'] > 25:
                    red.append([c, 'red'])
                    cv2.circle(blackimg, (c[0], c[1]), 3, (100, 100, 255), -1)
                # yellow
                elif color1['r'] - color1['b'] > 25:
                    yellow.append([c, 'yellow'])
                    cv2.circle(blackimg, (c[0], c[1]), 3, (100, 255, 255), -1)
                # green
                elif color1['g'] - color1['r'] > 25 and color1['b'] - color1['r'] > 25:
                    green.append([c, 'green'])
                    cv2.circle(blackimg, (c[0], c[1]), 3, (100, 255, 100), -1)
        self.colorpoint['blue'] = blue
        self.colorpoint['purple'] = purple
        self.colorpoint['white'] = white
        self.colorpoint['red'] = red
        self.colorpoint['yellow'] = yellow
        self.colorpoint['green'] = green
        if screen:
            cv2.imshow('sample', blackimg)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if output:
            cv2.imwrite(ResultImgPath+'getcolor'+str(self.alpha)+'_'+str(self.beta)+'_'+str(self.gama)+'.bmp', blackimg)
        return blackimg

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
                    # blackimg = cv2.imread('black1.bmp', 0)
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
                    mark = [index, i]   # TODO:Ask penguin
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
        return img2

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

    def SeeSee(self):
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
                cv2.circle(fig, (p[0], p[1]), 15, (0, 0, 255), 2)
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
        return fig
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 12:48:22 2018
@author: lj
"""
import tensorflow as tf
import numpy as np
import cv2 as cv
from scipy import *
from scipy.linalg import norm, pinv
from matplotlib import pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance
from sklearn.preprocessing import Imputer

TxTFileName = 'E://batch mark result.txt'
FigFolderName = 'E://imgimp//imim//'

def calculateH(figname):
    figname = FigFolderName+figname
    head, alpha, beta, gamaend = figname.split('_')
    gama, end = gamaend.split('.')
    # 转整数
    alpha = int(alpha)
    beta = int(beta)
    gama = int(gama)

    fig = cv.imread(figname)
    cv.imshow('original fig',fig)
    # 原图中的4个点
    src_point = np.float32([[50,100],[150,100],[50,200],[150,200]])
    dst_point = np.float32([[0,0],[1000,0],[0,1000],[1000,1000]])
    # 至少要4个点，一一对应，找到映射矩阵h
    h, s = cv.findHomography(src_point, dst_point, cv.RANSAC, 10)
    # test
    h = np.mat(h)
    o = np.mat([[150],[200],[1]])
    res = h*o
    res = np.asarray(res)
    res = res.astype(np.int32)
    return h






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


if __name__ == '__main__':
    n = 100
    x = mgrid[-1:1:complex(0, n)].reshape(n, 1)
    x2 = mgrid[-1:1:complex(0, n)].reshape(n, 1)
    xx = np.dstack((x, x2))
    # set y and add random noise
    y = sin(3 * (x + 0.5) ** 3 - 1)
    # x = [x, x2]
    # y += random.normal(0, 0.1, y.shape)

    # rbf regression
    rbf = RBF(2, 20, 1)
    rbf.train(xx, y)
    z = rbf.test(xx)

    # plot original data
    plt.figure(figsize=(12, 8))
    plt.plot(x, y, 'k-')

    # plot learned model
    plt.plot(x, z, 'r-', linewidth=2)
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
    

'''
if __name__ == '__main__':
    trainFilePath = 'dataset/soccer/train.csv'
    testFilePath = 'dataset/soccer/test.csv'
    data = loadDataset(trainFilePath)
    X_train, x_train, y_train = featureSet(data)
    X_test = loadTestData(testFilePath)
    trainandTest(X_train, y_train, X_test)'''
#if __name__ == '__main__':
   #calculateH('image_103_3_153.bmp')
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import operator
import time


a = np.loadtxt('PlaData.txt')

cnt=0

x0=np.ones([400,1])

trainData=np.c_[x0,a]

label=trainData[:,5]

trainData2=trainData[:,0:5]

def sigmoid(X):
    X=float(X)
    if X>0:
        return 1
    elif X<0:
        return -1
    else:
        return -1

def pla(traindataIn,trainlabelIn):
    traindata=np.mat(traindataIn)#将数组转化为矩阵matrix
    trainlabel=np.mat(trainlabelIn).transpose()#矩阵转置
    m,n=np.shape(traindata)
    w=np.zeros((n,1))
    global cnt
    while True:
        
        iscompleted=True
        for i in range(m):
            
            if (sigmoid(np.dot(traindata[i],w))==trainlabel[i]):#dot为矩阵乘法，*为对应元素相乘
                continue
            else:
                cnt=cnt+1
                iscompleted=False
                w+=(trainlabel[i]*traindata[i]).transpose()#*是指对应元素的相乘
        if iscompleted:
            print("cnt:%s" % cnt)
            break
    return w

def main():
    w=pla(trainData2,label)


if __name__=='__main__':
    start = time.clock()
    main()
    end = time.clock()
    print('finish all in %s' % str(end - start))



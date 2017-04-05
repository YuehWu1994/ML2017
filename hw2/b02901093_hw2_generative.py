#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sys
import csv
from numpy import genfromtxt
from numpy.linalg import inv
rootPath = ''
# global param
arg = False
selectFea = []
feaRow = 0
feaCol = 0
unnesFeature = np.asarray([])
#14,52,105

# age:0, fnlwgt:1, sex:2, capital_gain:3, capital_loss:4, hour_per_week:5
# workClass: 6~13
# 14: unknown
# Education 15~30
# Marital-Status: 31~37
# Occupation 38~51
# 52: unknown
# Relationship : 53~58
# Race: 59~63
# Nationalism 64~104
# 105 unknown


# var
# (wanted) 0,2,3,5,33,41,47,53
# (wanted) 0,2,3,4,5,24,27,29,33,41,47,53
# (wanted) 0,2,3,4,5,6,7,10,11,12,22,23,24,25,27,29,32,33,41,47,48,49,50,53,58,63,83,85,87
# (wanted) 0,2,3,4,5,24,27,29,33,35,41,47,53,54,56

# feaVar
# (wanted) > 0.5 : 0, 2, 3, 5, 33, 35, 41, 53, 56 
# (wanted) > 0.28 : 0, 2, 3,4, 5,10,24,25,26,27,29,31, 33, 35, 41,45,47, 53,54,56,57,58

# positive
# 3: capital_gain, 10: self-inc, 24: BS,25: PHD ,27: MS, 33: civ-Spouse, 40: craft-repair, 63: white,
# negative
# 35: never-married, 42: farming-fishing, 43: handlers-cleaner, 44: machine-op-inspect, 56: own-child, 57: unmarried,
# education: 15~21, 28
# all: 3,10,24,25,27,33,40,63,35,42,43,44,56,57,15,16,17,18,19,20,21,28
# continuous : 0,1,3,4,5

def seperateRaw(rawData):
    col = rawData[0].size
    unnesCol = unnesFeature.size
    selectFea = []
    for i in range(0,col):
        flag = True
        for j in range(0,unnesCol):
            if i == unnesFeature[j]:
                flag = False
        if flag == True:
            selectFea.append(rawData[:,i])     
    return np.asfarray(selectFea).transpose()

def stdScalar():
    std = np.std(selectFea,axis=0)
    mean = np.mean(selectFea,axis=0)
    for i in range(0,feaCol):
        if std[i] == 0:
            selectFea[:,i] = np.zeros((feaRow))  
        else:
            selectFea[:,i] = (selectFea[:,i] - mean[i])/std[i]

def countStdVector(ttl, idx, mean):
    tmp =[]
    for i in range(0,ttl):
        t = selectFea[idx[i]] - mean
        tt = t.transpose()
        if i == 0:
            tmp = np.dot(tt,t)              # not sure
        else:
            tmp = np.dot(tt,t) + tmp 
    tmp = tmp/ttl
    return tmp

def accCount(row,weightSet,bias):
    accCount = 0
    for i in range(0, row):
        predict = np.dot(selectFea[i], weightSet[0:feaCol]) + bias
        if (predict > 0 and rawY[i] == 0) or (predict <= 0 and rawY[i] == 1):
            accCount +=1
    print(accCount/row)
    return (accCount/row)

def countCcoef():
    ccfSet = np.zeros((feaCol,1))
    for i in range(0, feaCol):
      ccfSet[i] = np.corrcoef(selectFea[:,i],rawY)[0,1]
    return ccfSet   

def createFeature():
    addFeature(np.asarray([0,0,0]))
    addFeature(np.asarray([1,1,1]))
    addFeature(np.asarray([3,3,3]))
    #addFeature(np.asarray([4,4,4]))
    #addFeature(np.asarray([5,5,5]))
    
def addFeature(feaIdx):
    colFea = selectFea[:,0]
    for i in range(0,feaIdx.size-1):
        colFea = colFea + selectFea[:,i+1]
    global selectFea
    selectFea = np.concatenate((selectFea, np.reshape(colFea, (selectFea[:,0].size, 1))), axis=1)
    
# readFile     
if arg == True:
    df = pd.read_csv(sys.argv[3],encoding="big5")
else:
    df = pd.read_csv(rootPath + 'X_train.csv',encoding="big5")
rawData = df.as_matrix()

if arg == True:
    dfY = genfromtxt(sys.argv[4],dtype=None,delimiter=',')
else:
    rawY = genfromtxt('Y_train',dtype=None,delimiter=',')     
# paramSetting
selectFea = seperateRaw(rawData)
# continuous
#selectFea[:,0] = selectFea[:,0]**3
#selectFea[:,1] = selectFea[:,1]**3
#selectFea[:,3] = selectFea[:,3]**3
#selectFea[:,4] = selectFea[:,4]**3
#selectFea[:,5] = selectFea[:,5]**3
#createFeature()

feaRow = selectFea[:,0].size
feaCol = selectFea[0].size 
del df
del rawData
ccfSet = countCcoef()
stdScalar()
# get mean, std, weight, bias
ttl_1 = np.count_nonzero(rawY)
ttl_0 = rawY.size - ttl_1
idx_0 = np.where(rawY == 0)[0]
idx_1 = np.where(rawY == 1)[0]
mean0 = np.reshape(np.mean(selectFea[idx_0],axis = 0),(1,feaCol))  # not sure
mean1 = np.reshape(np.mean(selectFea[idx_1],axis = 0),(1,feaCol))
std_0 = countStdVector(ttl_0, idx_0, mean0)
std_1 = countStdVector(ttl_1, idx_1, mean1)
coStd = std_0 * (ttl_0/feaRow) + std_1 * (ttl_1/feaRow)
#weight = np.dot((mean0 - mean1).transpose(),inv(coStd)) 
weight = np.dot((mean0 - mean1),inv(coStd)) 
bias = float(-0.5*(np.dot(np.dot(mean0,inv(coStd)),mean0.transpose())) + 0.5*(np.dot(np.dot(mean1,inv(coStd)),mean1.transpose()))+np.log(ttl_0/ttl_1))

# feaDiff
diffSet = np.abs(np.sum(selectFea[idx_0,:],axis= 0)*(ttl_1/ttl_0) - np.sum(selectFea[idx_1,:],axis= 0))/ttl_1
acc = accCount(feaRow, np.reshape(weight,(feaCol,1)),bias)

# testing
if arg == True:
    predictData = pd.read_csv(sys.argv[5],encoding="big5")
else:
    predictData = pd.read_csv(rootPath + 'X_test',encoding="big5")
predictData = predictData.as_matrix()
selectTest = seperateRaw(predictData)
del predictData
feaRow = predictRow = selectTest[:,0].size
selectFea = selectTest


stdScalar()
parse = []

parse.append(['id','label'])
for k in range(0,predictRow):
    tmp = float(np.dot(weight,np.reshape((selectTest[k]),(feaCol,1)))) + bias  
    if tmp > 0:
        parse.append([str(k+1),0])
    else:
        parse.append([str(k+1),1])
if arg == True:
    pth = sys.argv[6]
else:
    pth = rootPath + 'genData.csv'
with open (pth, 'w') as f:
    writer = csv.writer(f)
    for k in range(0,(predictRow+1)):
        writer.writerow(parse[k])
        
'''        
parse = []
s = len(weight)
for k in range(0,s):
    parse.append([weight[k]])
parse.append(bias)
with open (rootPath +'genSubmit.csv', 'w') as f:
    writer = csv.writer(f)      
    for k in range(0,s+1):
        writer.writerow(parse[k])  
'''

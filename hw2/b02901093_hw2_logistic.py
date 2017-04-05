#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sys
import csv
import random
from numpy import genfromtxt
rootPath = ''
# global param
arg = False
selectFea = []
rawY = []
feaRow = 0
feaCol = 0
unnesFeature = np.asarray([])
rmsProp = np.asarray([])
learnRate = 10**-5
lamda = 0.08
alpha = 0.7
momentum = 0.3
models = 5000
accSet = np.zeros((1,models))
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
        selectFea[:,i] = (selectFea[:,i] - mean[i])/std[i] 

def trainTestSep(piv):
    if piv == 0:
        trainSet = selectFea[int(feaRow/3):,:]
        testSet = selectFea[0:int(feaRow/3),:]
        trainY = rawY[int(feaRow/3):]
        testY = rawY[0:int(feaRow/3)]    
    elif piv == 2:
        trainSet = selectFea[0:int(feaRow * 2/3),:]
        testSet = selectFea[int(feaRow * 2/3):,:]
        trainY = rawY[0:int(feaRow * 2/3)]
        testY = rawY[int(feaRow * 2/3):]
    elif piv == 1:
        trainSet = np.append(selectFea[0:int(feaRow/3),:],selectFea[int(feaRow * 2/3):,:], axis=0)   
        testSet = selectFea[int(feaRow * 1/3):int(feaRow * 2/3),:]
        trainY = rawY[int(feaRow * 1/3):int(feaRow * 2/3)]   
        testY = np.append(rawY[0:int(feaRow/3)],rawY[int(feaRow * 2/3):], axis=0)
    else:
        trainSet = selectFea
        testSet = []
        trainY = rawY
        testY = []            
    trainRow = trainSet[:,0].size
    if piv <= 2:
        testRow = testSet[:,0].size
    else :
        testRow = 0
    return trainSet, testSet, trainRow, testRow, trainY, testY

def genRandom():
    randSet = []
    for i in range(0, feaCol):
        randSet.append(float(random.uniform(-0.1,0.1))) 
    randSet.append(float(random.uniform(-1,1))) 
    return np.asarray(randSet[0:feaCol+1])

def weightChange(weightSet, trainSet, trainRow, testSet, rmsPropFlag, mmt):
    diffSet = []
    for i in range(0, trainRow):
        diffSet.append(trainY[i] - sigmoid(weightSet, trainSet[i]))
    diffSet = np.asarray(diffSet)
    # count loss
    lost = np.sum(np.abs(diffSet)) # not sure
    #print(lost,end=" ")
    nextWeight = np.zeros((feaCol+1))
    tmpWht = []
    for i in range(0, feaCol):
        tmpWht.append(np.sum(-1*(diffSet * trainSet[:,i])))
    tmpWht.append(np.sum(-1*diffSet))
    tmpWht = np.asarray(tmpWht)
    tmpWht[0:feaCol] = tmpWht[0:feaCol] + weightSet[0:feaCol]*lamda
    global rmsProp
    if rmsPropFlag == True:
        rmsProp = tmpWht
        rmsPropFlag = False
    rmsProp = np.sqrt(alpha * rmsProp**2 + (1-alpha) * tmpWht**2)
    mmt =  -(learnRate/rmsProp) * tmpWht + momentum * mmt
    nextWeight = weightSet + mmt
    return nextWeight, lost, rmsPropFlag, mmt

def sigmoid(weightSet, x):
    z = np.dot(weightSet[0:feaCol], x) + float(weightSet[feaCol])
    return (1/(1+np.exp(-z)))

def setStop(prevLost, nxLost, count):
    if nxLost > prevLost:
        count += 1
    else:
        count = 0
    prevLost = nxLost
    return prevLost, count    

def accCount(row,weightSet):
    accCount = 0
    for i in range(0, row):
        predict = np.dot(trainSet[i], weightSet[0:feaCol]) + float(weightSet[feaCol])
        if (predict >= 0 and trainY[i] == 1) or (predict < 0 and trainY[i] == 0):
            accCount +=1
    return (accCount/row)

def accValid(row,weightSet):
    accCount = 0
    for i in range(row-6000, row):
        predict = np.dot(trainSet[i], weightSet[0:feaCol]) + float(weightSet[feaCol])
        if (predict >= 0 and trainY[i] == 1) or (predict < 0 and trainY[i] == 0):
            accCount +=1
    return (accCount/6000)

def writeWeight(m,wantedW):
    print(m)
    for i in range (0, feaCol+1):
        print (str(wantedW[i]) + ',')
    print('\n')
    parse = []
    s = len(wantedW)
    for k in range(0,s):
        parse.append([wantedW[k]])
    with open (rootPath + str(accValid(trainRow,weightSet)) + '.csv', 'w') as f:
        writer = csv.writer(f)      
        for k in range(0,s):
            writer.writerow(parse[k])  

def createFeature():
    addFeature(np.asarray([1,1,1]))
    
    
def addFeature(feaIdx):
    colFea = selectFea[:,feaIdx[0]]
    for i in range(0,feaIdx.size-1):
        colFea = colFea * selectFea[:,feaIdx[i]]
    global selectFea
    selectFea = np.concatenate((selectFea, np.reshape(colFea, (selectFea[:,0].size, 1))), axis=1)

# readFile
     
if arg == True:
    df = pd.read_csv(sys.argv[3],encoding="big5")
else:
    df = pd.read_csv(rootPath + 'X_train.csv',encoding="big5")
rawData = df.as_matrix()

if arg == True:
    rawY = genfromtxt(sys.argv[4],dtype=None,delimiter=',')
else:
    rawY = genfromtxt('Y_train',dtype=None,delimiter=',')     
# paramSetting
selectFea = seperateRaw(rawData)
#**2
selectFea = np.concatenate((selectFea, selectFea[:,[0,1,3,4,5]]**3),axis = 1)
selectFea = np.concatenate((selectFea, selectFea[:,[0,1,3,4,5]]**2),axis = 1)
#createFeature()
feaRow = selectFea[:,0].size
feaCol = selectFea[0].size 
del df
del rawData
stdScalar()
#continuous:
#selectFea[:,0] = selectFea[:,0]**2
#selectFea[:,1] = selectFea[:,1]**2
#selectFea[:,3] = selectFea[:,3]**2
#selectFea[:,4] = selectFea[:,4]**2
#selectFea[:,5] = selectFea[:,5]**2

for m in range(0, models):
    weightSet = genRandom()
    rmsPropFlag = True
    trainSet, testSet, trainRow, testRow, trainY, testY = trainTestSep(3)
    prevLost = 10**9
    count = 0
    accFlag = True
    mmt = np.zeros((feaCol+1))
    while True:
        weightSet, nxLost, rmsPropFlag, mmt = weightChange(weightSet, trainSet, trainRow, testSet, rmsPropFlag, mmt)
        # difference threshold
        if np.abs(prevLost - nxLost) < 10**(-4): 
            acc = accValid(trainRow,weightSet)
            if acc > 0.85:
                print('stop by difference threshold: ')
                print(acc)
                writeWeight(m,weightSet)  
            else:
                print('not pass validation accuracy : ' + str(acc))
            break
        prevLost, count = setStop(prevLost,nxLost, count)
        # init Acc threshold
        if accFlag == True:
            if accCount(trainRow,weightSet) < 0.75:
                accSet[0,m] = accCount(trainRow,weightSet)
                print('not pass Acc threshold', end = ' ')
                print(accSet[0,m])
                break
            else:
                print('pass')
                print(accCount(trainRow,weightSet))
                accFlag = False
        else:
            print(accCount(trainRow,weightSet))
            '''
            for i in range (0, feaCol+1):
                print (str(weightSet[i]) + ',', end = '')
            print('\n')
            '''
        # lost increase threshold
        if count == 3:
            acc = accValid(trainRow,weightSet)
            if acc > 0.85:
                print('stop by count: ')
                print(acc)
                writeWeight(m,weightSet)
            else:
                print('not pass validation accuracy : ' + str(acc))
            break

"""
for sep in range(0,3):
    trainSet, testSet, trainRow, testRow, trainY, testY = trainTestSep(sep)
    prevLost = 10**8
    count = 0
    while True:
        weightSet, nxLost = weightChange(weightSet, trainSet, trainRow, testSet, bias)
        prevLost, count = setStop(prevLost,nxLost, count)
        print(count)
        if count == 3:
            acc = accCount(testRow,weightSet)           
            break
"""



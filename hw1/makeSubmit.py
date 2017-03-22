#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import sys
import csv
from numpy import genfromtxt
numFeature = 18
arg = True

def feaSelect(featureRow, trainFea, ans):
    numFea = 1
    tmpSet = np.zeros((featureRow, int(trainFea/numFeature) * numFea + 1))
    pm2_5 = 9
    for i in range(0, featureRow):
         tmpSet[i,0:9:1] = featureSet[i][pm2_5:162:18]
         tmpSet[i,9] = featureSet[i][ans]
    return tmpSet

def sepFeatureTest(row,column,data):
    days = int(row/numFeature)
    for d in range(0, days):
        for c in range(2, column):
            for i in range(0, numFeature):
                tmp = data[d * numFeature + i][c]
                if i == 10:
                    tmp = 0
                featureSet.append(float(tmp))

weight = genfromtxt("model.csv",dtype=None,delimiter=',')

featureSet = []
if arg == True:
    testData = genfromtxt(sys.argv[2],dtype=None,delimiter=',')
else:
    testData = genfromtxt("/Users/apple/desktop/NTUEE/課程八/ML/git/hw1/test_X.csv",dtype=None,delimiter=',')    
   
column = int(testData[0].size)
row = len(testData)
sepFeatureTest(row,column,testData)
del testData
trainFea = 162   
featureSet = np.reshape(np.asarray(featureSet),(-1,trainFea)) 
featureRow = int(featureSet[:,0].size)

featureSet = feaSelect(featureRow, trainFea, 9)
ans = featureSet[0].size -1
trainFea = ans

predict = []
for r in range (0,featureRow):
    estAns = 0
    for i in range(0, 9):
        estAns += featureSet[r,i] * weight[i]
    estAns += weight[9]
    predict.append(estAns)
parse = []
parse.append(['id','value'])
for k in range(0,240):
    parse.append(['id_' + str(k), predict[k]])


if arg == True:
    pth = sys.argv[3]
else:
    pth = "/Users/apple/desktop/NTUEE/課程八/ML/git/hw1/submi1.csv"
with open (pth, 'w') as f:
    writer = csv.writer(f)
    for k in range(0,241):
        writer.writerow(parse[k])


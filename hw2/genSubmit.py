#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sys
import csv
from numpy import genfromtxt
rootPath = ''
arg = True
selectFea = []
feaRow = 0
feaCol = 0
def stdScalar():
    std = np.std(selectFea,axis=0)
    mean = np.mean(selectFea,axis=0)
    for i in range(0,feaCol):
        if std[i] == 0:
            selectFea[:,i] = np.zeros((feaRow))  
        else:
            selectFea[:,i] = (selectFea[:,i] - mean[i])/std[i]

# readFile     
if arg == True:
    df = pd.read_csv(sys.argv[5],encoding="big5")
else:
    df = pd.read_csv(rootPath + 'X_test',encoding="big5")
selectFea = df.as_matrix()
selectFea = selectFea.astype(float)
feaRow = selectFea[:,0].size
feaCol = selectFea[0].size 
del df
stdScalar()
weight = np.reshape(genfromtxt("modelGen.csv",dtype=None,delimiter=','),(107,2))
bias = float(weight[106,0])
predict = np.dot(selectFea, weight[0:106,0])+   + bias

parse = []
parse.append(['id','label'])
for k in range(0,feaRow):  
    if predict[k] > 0:
        parse.append([str(k+1),0])
    else:
        parse.append([str(k+1),1])
if arg == True:
    pth = sys.argv[6]
else:
    pth = rootPath + 'generative.csv'
with open (pth, 'w') as f:
    writer = csv.writer(f)
    for k in range(0,(feaRow+1)):
        writer.writerow(parse[k])

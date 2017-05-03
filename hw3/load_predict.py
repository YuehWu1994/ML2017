#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sys
import csv
from numpy import genfromtxt
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten, ZeroPadding2D
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop
#rootPath = '/Users/apple/desktop/NTUEE/課程八/ML/git/hw3/'
model2 = load_model("0.67679.h5")
arg = True

# testing
if arg == True:
    df = pd.read_csv(sys.argv[1],encoding="big5") 
else:
    df = pd.read_csv(rootPath + 'test.csv',encoding="big5")    
rawData = df.as_matrix()
dfS = len(df)
testX = [[] for _ in range(dfS)]
for i in range(0, dfS):
    testX[i] = np.asarray(list(map(int,rawData[i,1].split(' '))))
testX = np.asarray(testX).astype(float)/255
testX = testX.reshape(testX.shape[0],48,48,1)
del df, rawData
pdWeight = model2.predict(testX)
predictX = np.argmax(pdWeight, axis = 1)
        
# writeOutput
parse = []
parse.append(['id','label'])
row = predictX.size
for k in range(0,row):  
    parse.append([str(k),predictX[k]])
    
if arg == True:
    testFile = sys.argv[2]
else:
    testFile = 'testP.txt'
with open (testFile, 'w') as f:
    writer = csv.writer(f)
    for k in range(0,row+1):
        writer.writerow(parse[k])        

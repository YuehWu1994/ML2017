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
rootPath = '/home/bl418/桌面/093/'
#rootPath = '/home/pc193/093/'
print('param : 1actFunc, 2#ofHiddenLayer, 3#ofHiddenUnit, 4batchSize, 5#epoch, 6optimizer(ex: adam, SGD), 7BatchNorm, 8filterR, 9filterR2, 10dropout, 11dropout2, 12modelNum, 13conv2D, 14conv2D2, 15BN2, 16cov2D3, 17filter3, 18cov2D4, 19filter4')
# parameter
arg = True
modelNum = 10
def addZero(addToLabel):
    r = len(addToLabel)
    y = np.zeros((r,7))
    for i in range(0,r):
        y[i,addToLabel[i,1]] = 1
    return y

if arg == True:
    act = sys.argv[1]
    hidLayer = int(sys.argv[2])
    hidUnit = int(sys.argv[3])
    batchSize = int(sys.argv[4])
    epoch = int(sys.argv[5])
    opt = sys.argv[6]
    if opt != 'adam':
        opt = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
    BN = int(sys.argv[7])
    fltR = int(sys.argv[8])
    fltR2 = int(sys.argv[9])
    dp = float(sys.argv[10])
    dp2 = float(sys.argv[11])
    modelNum = int(sys.argv[12])
    cov2D = int(sys.argv[13])
    cov2D2 = int(sys.argv[14])
    BN2 = int(sys.argv[15])
    cov2D3 = int(sys.argv[16])
    fltR3 = int(sys.argv[17])
    cov2D4 = int(sys.argv[18])
    fltR4 = int(sys.argv[19])
    semiInitRow = int(sys.argv[20])
    semiEpoch = int(sys.argv[21])
    thr = float(sys.argv[22])
else:
    act = 'elu'
    hidLayer = 2
    hidUnit = 256
    batchSize = 100
    epoch = 1
    opt = 'adam'
    if opt != 'adam':
        opt = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
    BN = 0
    fltR = 3
    fltR2 = 3   
    dp = 0.25
    dp2 = 0.5
    modelNum = 10
    cov2D = 25
    cov2D2 = 50
    BN2 = 0
    cov2D3 = 0
    fltR3 = 3
    cov2D4 = 0
    fltR4 = 3
    semiInitRow = 5000
    semiEpoch = 1
    thr = 0.9
# init data
df = pd.read_csv(rootPath + 'train.csv',encoding="big5")
rawData = df.as_matrix()
dfS = len(df)
feaX = [[] for _ in range(dfS)]
feaY = np.zeros((dfS,7))
for i in range(0, dfS):
    feaX[i] = np.asarray(list(map(int,rawData[i,1].split(' '))))
    feaY[i,int(rawData[i,0])] = 1
feaX = np.asarray(feaX).astype(float)/255
del df, rawData

df = pd.read_csv(rootPath + 'test.csv',encoding="big5")
rawData = df.as_matrix()
dfSt = len(df)
testX = [[] for _ in range(dfSt)]
for i in range(0, dfSt):
    testX[i] = np.asarray(list(map(int,rawData[i,1].split(' '))))
testX = np.asarray(testX).astype(float)/255
testX = testX.reshape(testX.shape[0],48,48,1)
del df, rawData

# sep training & testing in training set
aveacc = []
for i in range(0,1):
    # VGG
    model2 = Sequential()
    model2.add(ZeroPadding2D((1,1),input_shape = (48,48,1)))
    model2.add(Conv2D(cov2D,(fltR,fltR)))
    if BN2 == 1:
        model2.add(BatchNormalization())
    model2.add(Activation(act))
    
    model2.add(ZeroPadding2D((1,1)))
    model2.add(Conv2D(cov2D,(fltR,fltR)))
    if BN2 == 1:
        model2.add(BatchNormalization())
    model2.add(Activation(act))
    
    model2.add(MaxPooling2D((2,2)))
    model2.add(Dropout(dp))
            
    model2.add(ZeroPadding2D((1,1)))
    model2.add(Conv2D(cov2D2,(fltR2,fltR2)))
    if BN2 == 1:
    		model2.add(BatchNormalization())    
    model2.add(Activation(act))
    
    model2.add(ZeroPadding2D((1,1)))
    model2.add(Conv2D(cov2D2,(fltR2,fltR2)))
    if BN2 == 1:
    		model2.add(BatchNormalization())
    model2.add(Activation(act))
    
    model2.add(MaxPooling2D((2,2)))
    model2.add(Dropout(dp))
    
    if cov2D3 != 0:
        model2.add(ZeroPadding2D((1,1)))
        model2.add(Conv2D(cov2D3,(fltR3,fltR3)))
        if BN2 == 1:
        		model2.add(BatchNormalization())    
        model2.add(Activation(act))
        
        model2.add(ZeroPadding2D((1,1)))
        model2.add(Conv2D(cov2D3,(fltR3,fltR3)))
        if BN2 == 1:
        		model2.add(BatchNormalization())
        model2.add(Activation(act))
        
        model2.add(MaxPooling2D((2,2)))
        model2.add(Dropout(dp))

    if cov2D4 != 0:
        model2.add(ZeroPadding2D((1,1)))
        model2.add(Conv2D(cov2D4,(fltR4,fltR4)))
        if BN2 == 1:
        		model2.add(BatchNormalization())    
        model2.add(Activation(act))
        
        model2.add(ZeroPadding2D((1,1)))
        model2.add(Conv2D(cov2D4,(fltR4,fltR4)))
        if BN2 == 1:
        		model2.add(BatchNormalization())
        model2.add(Activation(act))
        
        model2.add(MaxPooling2D((2,2)))
        model2.add(Dropout(dp))
    
    model2.add(Flatten())
    # Dense
    for j in range(0,hidLayer):
        model2.add(Dense(units=hidUnit))
        if BN == 1:
            model2.add(BatchNormalization())
        model2.add(Activation(act))
        model2.add(Dropout(dp2))
    # Output
    model2.add(Dense(units=7))
    if BN == 1:
        model2.add(BatchNormalization())
    model2.add(Activation('softmax'))
    model2.summary()
    
    x_test = feaX[int(np.ceil(dfS/modelNum*i)):int(np.ceil(dfS/modelNum*(i+1)))]
    y_test = feaY[int(np.ceil(dfS/modelNum*i)):int(np.ceil(dfS/modelNum*(i+1)))]
    x_train = np.concatenate((feaX[0:int(np.ceil(dfS/modelNum*i))], feaX[int(np.ceil(dfS/modelNum*(i+1))):]), axis=0)
    y_train = np.concatenate((feaY[0:int(np.ceil(dfS/modelNum*i))], feaY[int(np.ceil(dfS/modelNum*(i+1))):]), axis=0) 

    # semi-Init
    x_semiTrain = x_train
    x_train = testX
    y_semiTrain = y_train
    x_semiTrain = x_semiTrain.reshape(x_semiTrain.shape[0],48,48,1)
    x_test = x_test.reshape(x_test.shape[0],48,48,1)
    unlabelRow = len(x_train)
    t = 0
    
    while (unlabelRow > 0 and t < 20):
        print ('# of Unlabel Row is :' + str(unlabelRow) + ', repete ' + str(t) + 'times' )
        model2.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
        model2.fit(x_semiTrain,y_semiTrain,batch_size=batchSize,epochs=semiEpoch,validation_data=(x_test, y_test))
        addToLabel = []
        pdWeight = model2.predict(x_train)
        pdRow = len(pdWeight)
        for j in range(0,pdRow):
            c = np.where(pdWeight[j] > thr)[0]
            if len(c) > 0:
                cls = c[0]
                addToLabel.append([j,cls])
        addToLabel = np.asarray(addToLabel)
        if len(addToLabel) > 0:
            x_semiTrain =np.concatenate((x_semiTrain,x_train[addToLabel[:,0]]),axis = 0)
            yP = addZero(addToLabel)
            y_semiTrain =np.concatenate((y_semiTrain,yP),axis = 0)
            x_train = np.delete(x_train, addToLabel[:,0], 0)
            unlabelRow = unlabelRow - len(addToLabel)
        t = t+1
                   
    score = model2.evaluate(x_semiTrain,y_semiTrain)
    print ('\nTrain Acc:', score[1])
    scoreTest = model2.evaluate(x_test,y_test)
    print ('\nTest Acc:', scoreTest[1])
    aveacc.append([score[1], scoreTest[1]])
    # save & load
    model2.save(str(i)+ "model.h5")
    del model2


aveacc = np.asarray(aveacc)
idd = np.argmax(np.sum(aveacc, axis = 1),axis = 0)
model2 = load_model(str(idd)+ "model.h5")
print('Opt Model is index' + str(idd) + '.  Ave Testing Acc is : ' + str(np.mean(aveacc[:,1])))    
with open(str(aveacc[idd,1]) + '_' + str(act) + '_' + str(hidLayer) + '_' + str(hidUnit) + '_' + str(batchSize)+ '_' + str(epoch) + '_' + str(opt) + '_' + str(BN) + '_' + str(fltR)+ '_' + str(fltR2) +'_' + str(modelNum) + '_' + str(cov2D) + '_' + str(cov2D2) + '_' + str(BN2) + '.csv', 'w') as f: 
    f.write(str(np.max(aveacc[:,1])))
    f.close()
    
# testing
pdWeight = model2.predict(testX)
predictX = np.argmax(pdWeight, axis = 1)
model2.save(rootPath + str(aveacc[idd,1]) +'.h5')

        
# writeOutput
parse = []
parse.append(['id','label'])
row = predictX.size
for k in range(0,row):  
    parse.append([str(k),predictX[k]])
pth = rootPath + str(aveacc[idd,1]) + '_' + str(act) + '_' + str(hidLayer) + '_' + str(hidUnit) + '_' + str(batchSize)+ '_' + str(epoch) + '_' + str(BN) + '_' + str(fltR)+ '_' + str(fltR2) +'_' + str(modelNum) + '_' + str(cov2D) + '_' + str(cov2D2) + '_' + str(BN2) + '_' + str(cov2D3) + '_' + str(fltR3) + '_' + str(cov2D4) + '_' + str(fltR4) + '.csv'
with open (pth, 'w') as f:
    writer = csv.writer(f)
    for k in range(0,row+1):
        writer.writerow(parse[k]) 

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
import matplotlib.pyplot as plt
#rootPath = '/home/pc193/093/'
rootPath = '/home/bl418/桌面/093/'
print('param : 1actFunc, 2#ofHiddenLayer, 3#ofHiddenUnit, 4batchSize, 5#epoch, 6optimizer(ex: adam, SGD), 7BatchNorm, 8filterR, 9filterR2, 10dropout, 11dropout2, 12modelNum, 13conv2D, 14conv2D2, 15BN2, 16conv2D3, 17filter3, 18conv2D4, 19filter4')
# parameter
arg = False
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
    x_train = x_train.reshape(x_train.shape[0],48,48,1)
    x_test = x_test.reshape(x_test.shape[0],48,48,1)
    model2.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
    hist = model2.fit(x_train,y_train,batch_size=batchSize,epochs=epoch,validation_data=(x_test, y_test))
    score = model2.evaluate(x_train,y_train)
    print ('\nTrain Acc:', score[1])
    scoreTest = model2.evaluate(x_test,y_test)
    print ('\nTest Acc:', scoreTest[1])
    aveacc.append([score[1], scoreTest[1]])
    # save & load
    model2.save(str(i)+ "model.h5")
    del model2
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('model_acc')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train','test'],loc='upper left')
    plt.show()
    plt.savefig('train_procedure.png')

    
aveacc = np.asarray(aveacc)
idd = np.argmax(np.sum(aveacc, axis = 1),axis = 0)
model2 = load_model(str(idd)+ "model.h5")
print('Opt Model is index' + str(idd) + '.  Ave Testing Acc is : ' + str(np.mean(aveacc[:,1])))    
with open(str(aveacc[idd,1]) + '_' + str(act) + '_' + str(hidLayer) + '_' + str(hidUnit) + '_' + str(batchSize)+ '_' + str(epoch) + '_' + str(BN) + '_' + str(fltR)+ '_' + str(fltR2) +'_' + str(modelNum) + '_' + str(cov2D) + '_' + str(cov2D2) + '_' + str(BN2) + '_' + str(cov2D3) + '_' + str(fltR3) + '_' + str(cov2D4) + '_' + str(fltR4) + '.csv', 'w') as f: 
    f.write(str(np.max(aveacc[:,1])))
    f.close()

#testing
df = pd.read_csv(rootPath + 'test.csv',encoding="big5")
rawData = df.as_matrix()
dfS = len(df)
testX = [[] for _ in range(dfS)]
for i in range(0, dfS):
    testX[i] = np.asarray(list(map(int,rawData[i,1].split(' '))))
testX = np.asarray(testX).astype(float)/255
testX = testX.reshape(testX.shape[0],48,48,1)
del df, rawData
predictX = np.argmax(model2.predict(testX), axis = 1)
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
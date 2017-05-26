#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sys
import csv
import keras
import keras.backend as K
from numpy import genfromtxt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers import  Dropout, Activation, TimeDistributed, Dense, Input
from keras.layers.recurrent import GRU, LSTM
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint


def unique_list(l):
    ulist = []
    [ulist.append(x) for x in l if x not in ulist]
    return ulist
"""
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fbeta_score(y_true, y_pred, beta=1):
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=1)
"""
###########################
###   custom metrices   ###
###########################
def f1_score(y_true,y_pred):
    thresh = 0.4
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true * y_pred,axis=-1)
    
    precision=tp/(K.sum(y_pred,axis=-1)+K.epsilon())
    recall=tp/(K.sum(y_true,axis=-1)+K.epsilon())
    return K.mean(2*((precision*recall)/(precision+recall+K.epsilon())))

rootPath = '/home/pc193/093/093Hw5/'
arg = False
debugArg = True
setVal = True
MAX_NB_WORDS = 51867
EMBEDDING_DIM = 100
MAX_SEQUENCE_LENGTH = 320
#LABEL_NB_WORDS = 43
#LABEL_SEQUENCE_LENGTH = 14
trainLine = []
testLine = []
label = []
testLabel = []
#f1score = fmeasure

if debugArg == True:
    batchSz = int(sys.argv[1])
    epoch = int(sys.argv[2])
    act = sys.argv[3]
    hidLayer = int(sys.argv[4])
    lay1 = int(sys.argv[5])
    lay2 = int(sys.argv[6])
    lay3 = int(sys.argv[7])
    dDp1 = float(sys.argv[8])
    dDp2 = float(sys.argv[9])
    dDp3 = float(sys.argv[10])
    actOut = sys.argv[11]

else:
    batchSz = 128
    epoch = 100
    act = 'relu'
    hidLayer = 3
    lay1 = 256
    lay2 = 128
    lay3 = 64
    dDp1 = 0.3
    dDp2 = 0.3
    dDp3 = 0.3
    actOut = 'sigmoid'

log = (str(batchSz) + '_' + str(epoch) + '_' + str(act) + '_'
      + str(hidLayer)+ '_'  + str(lay1) + '_'  + str(lay2) + '_'  + str(lay3) + '_'
      + str(dDp1)+ '_'  + str(dDp2) + '_'  + str(dDp3) + '_'  + actOut)
print (log)
# trainInit
if arg == True:
    trainPath = sys.argv[1]
else:
    trainPath = rootPath + 'train_data.csv'

with open(trainPath, 'r') as df:
    for l in df:
        spl = l.split(',')
        label.append(spl[1])     
        trainLine.append(','.join(spl[2:]))  
label = label[1:]
trainLine = trainLine[1:]
row = len(trainLine)
valId = int(row * 0.9)
# testInit
if arg == True:
    testPath = sys.argv[2]
else:
    testPath = rootPath + 'test_data.csv'
with open(testPath, 'r') as df:
    for l in df:
        spl = l.split(',') 
        testLine.append(','.join(spl[1:]))    
testLine = testLine[1:]
testRow = len(testLine)

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(trainLine + testLine)
data = tokenizer.texts_to_matrix(trainLine + testLine)
testData  = data[row:]
data = data[:row]
# 38
listLabel = ''
for i in range(0,row):
    listLabel = listLabel + label[i][1:len(label[i])-1] + ' '
listLabel = unique_list(listLabel.split())

labelData = np.zeros((row,38))
for i in range(0,row):
    cls = label[i][1:len(label[i])-1].split(' ')
    nbCls = len(cls)
    for j in range(0, nbCls):
        for k in range(0,38):
            if cls[j] == listLabel[k]:
                labelData[i][k] = 1
       

# set validation data or not
if setVal == True:
    trainDb = data[:valId]
    trainLabel = labelData[:valId]
    valDb = data[valId:]
    valLabel = labelData[valId:]
else:
    trainDb = data
    trainLabel = labelData
    
model1 = Sequential()
model1.add(Dense(lay1, input_shape=(MAX_NB_WORDS,)))
model1.add(BatchNormalization())
model1.add(Activation('relu'))
model1.add(Dropout(dDp1))

if hidLayer > 1:
    model1.add(Dense(lay2))
    model1.add(BatchNormalization())
    model1.add(Activation('relu'))
    model1.add(Dropout(dDp2))
if hidLayer > 2:
    model1.add(Dense(lay3))
    model1.add(BatchNormalization())
    model1.add(Activation('relu'))
    model1.add(Dropout(dDp3))



model1.add(Dense(38, activation='sigmoid'))
model1.summary()
model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1_score])
earlystopping = EarlyStopping(monitor='val_f1_score', patience = 20, verbose=1, mode='max')
checkpoint = ModelCheckpoint(filepath= (log +".hdf5"),
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 monitor='val_f1_score',
                                 mode='max')
model1.fit(trainDb, trainLabel,
          batch_size= batchSz,
          epochs=epoch,
          validation_data=(valDb, valLabel),
          callbacks=[earlystopping,checkpoint])

score, acc1 = model1.evaluate(valDb, valLabel,batch_size=128)
print('Test score:', score)
print('Test accuracy:', acc1)
model1.save(str(acc1) + '_' + log+ ".h5")
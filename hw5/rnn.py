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
    gruLayer = int(sys.argv[3])
    gru1U = int(sys.argv[4])
    dp1 = float(sys.argv[5])
    gru2U = int(sys.argv[6])
    dp2 = float(sys.argv[7])
    hidLayer = int(sys.argv[8])
    lay1 = int(sys.argv[9])
    lay2 = int(sys.argv[10])
    lay3 = int(sys.argv[11])
    dDp1 = float(sys.argv[12])
    dDp2 = float(sys.argv[13])
    dDp3 = float(sys.argv[14])
    actOut = sys.argv[15]
    gru3U = int(sys.argv[16])
    dp3 = float(sys.argv[17])

else:
    batchSz = 128
    epoch = 100
    gruLayer = 2
    gru1U = 128
    dp1 = 0.2
    gru2U = 64
    dp2 = 0.2
    hidLayer = 3
    lay1 = 256
    lay2 = 128
    lay3 = 64
    dDp1 = 0.2
    dDp2 = 0.2
    dDp3 = 0.2
    actOut = 'sigmoid'
    gru3U = 32
    dp3 = 0.2
   
log = (str(batchSz) + '_' + str(epoch) + '_' + str(gruLayer) + '_'  
      + str(gru1U)+ '_'  + str(dp1) + '_'  + str(gru2U) + '_'  + str(dp2) + '_' 
      + str(hidLayer)+ '_'  + str(lay1) + '_'  + str(lay2) + '_'  + str(lay3) + '_'
      + str(dDp1)+ '_'  + str(dDp2) + '_'  + str(dDp3) + '_'  + actOut+ str(gru3U)+ '_'
      + str(dp3)+'_EMBEDDING_DIM='+str(EMBEDDING_DIM))
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
sequences = tokenizer.texts_to_sequences(trainLine + testLine)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
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
    
   
# glove
embeddings_index = {}
f = open('glove.6B.'+ str(EMBEDDING_DIM) + 'd.txt', 'r')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()


# compute embedding matrix
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
"""
model2 = Sequential()
model2.add(embedding_layer)
model2.add(GRU(128, dropout=0.2, recurrent_dropout=0.2))
model2.add(Dense(38, activation='sigmoid'))
model2.summary()
model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1_score])
earlystopping = EarlyStopping(monitor='val_f1_score', patience = 10, verbose=1, mode='max')
checkpoint = ModelCheckpoint(filepath='logmodel2.hdf5',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 monitor='val_f1_score',
                                 mode='max')
model2.fit(trainDb, trainLabel,
          batch_size= 128,
          epochs=70,
          validation_data=(valDb, valLabel),
          callbacks=[earlystopping,checkpoint])

score, acc = model2.evaluate(valDb, valLabel,batch_size=128)
print('Test score:', score)
print('Test accuracy:', acc)
model2.save('70_128'+ "model2.h5")
"""

model = Sequential()
#model.add(Embedding(len(word_index) + 1, EMBEDDING_DIM))
model.add(embedding_layer)
if gruLayer >= 1:
    if gruLayer == 1:
        rSeq1 = False
    else: 
        rSeq1 = True
    model.add(GRU(gru1U, dropout=dp1, recurrent_dropout=dp1, return_sequences=rSeq1,activation='tanh'))

if gruLayer >=2:
    if gruLayer == 2:
        rSeq2 = False
    else: 
        rSeq2 = True
    model.add(GRU(gru2U, dropout=dp2, recurrent_dropout=dp2, return_sequences=rSeq2,activation='tanh'))

if gruLayer >=3:
    if gruLayer == 3:
        rSeq3 = False
    else: 
        rSeq3 = True
    model.add(GRU(gru3U, dropout=dp3, recurrent_dropout=dp3, return_sequences=rSeq3,activation='tanh'))   

for j in range(0,hidLayer):
    if j == 0:        
        model.add(Dense(lay1,activation='relu'))
        model.add(Dropout(dDp1))
    if j == 1:       
        model.add(Dense(lay2,activation='relu'))
        model.add(Dropout(dDp2))
    if j == 2:      
        model.add(Dense(lay3,activation='relu'))
        model.add(Dropout(dDp3))
    
model.add(Dense(38, activation=actOut))
model.summary()

# adam = Adam(lr=0.001,decay=1e-6,clipvalue=0.5)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=[f1_score])
earlystopping = EarlyStopping(monitor='val_f1_score', patience = 20, verbose=1, mode='max')
checkpoint = ModelCheckpoint(filepath= log +".hdf5",
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 monitor='val_f1_score',
                                 mode='max')
"""
model.fit(trainDb, trainLabel,
          batch_size= batchSz ,
          epochs=epoch,
          validation_data=(valDb, valLabel),
          callbacks=[earlystopping,checkpoint])
"""
#model.load_weights('xxx.hdf5')
score, acc = model.evaluate(valDb, valLabel,batch_size=batchSz)
print('Test score:', score)
print('Test accuracy:', acc)
model.save(str(acc) + log+ "model.h5")


"""
pred = model.predict(testData)
argmaxPred = np.argmax(pred, axis = 1)
for i in range(0,testRow):
    pred[i] = pred[i]*(1/pred[i,argmaxPred[i]])

    
pred2 = model2.predict(testData)
argmaxPred2 = np.argmax(pred2, axis = 1)
for i in range(0,testRow):
    pred2[i] = pred2[i]*(1/pred2[i,argmaxPred2[i]])
   

thr = 1
pth = rootPath + str(acc)+ '_' +str(thr) + '_' + log + '.csv'
f = open(pth, 'w')
f.write('"id","tags"\n')
for i in range(0,testRow):
    lb = '"'
    for j in range(0,38):       
        if (pred[i,j]+pred2[i,j]) >= thr:
            lb += (listLabel[j] + ' ')
    lb = lb[:len(lb)-1]
    lb += '"\n'
    f.write('"' + str(i) + '"'+','+ lb)
f.close()


thr = 1.2
pth = rootPath + str(acc)+ '_' +str(thr) + '_' + log + '.csv'
f = open(pth, 'w')
f.write('"id","tags"\n')
for i in range(0,testRow):
    lb = '"'
    for j in range(0,38):       
        if (pred[i,j]+pred2[i,j]) >= thr:
            lb += (listLabel[j] + ' ')
    lb = lb[:len(lb)-1]
    lb += '"\n'
    f.write('"' + str(i) + '"'+','+ lb)
f.close()

thr = 0.8
pth = rootPath + str(acc)+ '_' +str(thr) + '_' + log + '.csv'
f = open(pth, 'w')
f.write('"id","tags"\n')
for i in range(0,testRow):
    lb = '"'
    for j in range(0,38):       
        if (pred[i,j]+pred2[i,j]) >= thr:
            lb += (listLabel[j] + ' ')
    lb = lb[:len(lb)-1]
    lb += '"\n'
    f.write('"' + str(i) + '"'+','+ lb)
f.close()
"""

"""
with open (pth, 'w') as f:
    writer = csv.writer(f)
    for k in range(0,testRow+1):
        writer.writerow(parse[k])
"""



#ISO-8859-1

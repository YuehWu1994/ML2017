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
# model0
model = Sequential()
model.add(embedding_layer)

model.add(GRU(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True,activation='tanh'))
model.add(GRU(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=False,activation='tanh'))
       
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.2))     
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))    
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(38, activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=[f1_score])
earlystopping = EarlyStopping(monitor='val_f1_score', patience = 20, verbose=1, mode='max')
checkpoint = ModelCheckpoint(filepath="model.hdf5",
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 monitor='val_f1_score',
                                 mode='max')

model.fit(trainDb, trainLabel,
          batch_size= 128 ,
          epochs=100,
          validation_data=(valDb, valLabel),
          callbacks=[earlystopping,checkpoint])

score, acc = model.evaluate(valDb, valLabel,batch_size=128)
print('Test score:', score)
print('Test accuracy:', acc)
model.save('3_256'+ "model.h5")


# model 1
model1 = Sequential()
model1.add(embedding_layer)
model1.add(GRU(128, dropout=0.2, recurrent_dropout=0.2))
model1.add(Dense(38, activation='sigmoid'))
model1.summary()
model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1_score])
earlystopping = EarlyStopping(monitor='val_f1_score', patience = 20, verbose=1, mode='max')
checkpoint = ModelCheckpoint(filepath="model1.hdf5",
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 monitor='val_f1_score',
                                 mode='max')
model1.fit(trainDb, trainLabel,
          batch_size= 128,
          epochs=70,
          validation_data=(valDb, valLabel),
          callbacks=[earlystopping,checkpoint])

score, acc1 = model1.evaluate(valDb, valLabel,batch_size=128)
print('Test score:', score)
print('Test accuracy:', acc1)
model1.save('70_128_0.2'+ "model1.h5")

# model 2
model2 = Sequential()
model2.add(embedding_layer)
model2.add(GRU(256, dropout=0.2, recurrent_dropout=0.2))
model2.add(Dense(38, activation='sigmoid'))
model2.summary()
model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1_score])
earlystopping = EarlyStopping(monitor='val_f1_score', patience = 20, verbose=1, mode='max')
checkpoint = ModelCheckpoint(filepath="model2.hdf5",
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

score, acc2 = model2.evaluate(valDb, valLabel,batch_size=128)
print('Test score:', score)
print('Test accuracy:', acc2)
model2.save('70_256_0.2'+ "model2.h5")

# model 3
model3 = Sequential()
model3.add(embedding_layer)
model3.add(GRU(128, dropout=0.5, recurrent_dropout=0.2))
model3.add(Dense(38, activation='sigmoid'))
model3.summary()
model3.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1_score])
earlystopping = EarlyStopping(monitor='val_f1_score', patience = 20, verbose=1, mode='max')
checkpoint = ModelCheckpoint(filepath="model3.hdf5",
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 monitor='val_f1_score',
                                 mode='max')
model3.fit(trainDb, trainLabel,
          batch_size= 128,
          epochs=100,
          validation_data=(valDb, valLabel),
          callbacks=[earlystopping,checkpoint])

score, acc3 = model3.evaluate(valDb, valLabel,batch_size=128)
print('Test score:', score)
print('Test accuracy:', acc3)
model3.save('70_256_0.5'+ "model3.h5")

pred = model.predict(testData)
argmaxPred = np.argmax(pred, axis = 1)
for i in range(0,testRow):
    pred[i] = pred[i]*(1/pred[i,argmaxPred[i]])
    
pred1 = model1.predict(testData)
argmaxPred1 = np.argmax(pred1, axis = 1)
for i in range(0,testRow):
    pred1[i] = pred1[i]*(1/pred1[i,argmaxPred1[i]])

pred2 = model2.predict(testData)
argmaxPred2 = np.argmax(pred2, axis = 1)
for i in range(0,testRow):
    pred[i] = pred2[i]*(1/pred2[i,argmaxPred2[i]])
    
pred3 = model3.predict(testData)
argmaxPred3 = np.argmax(pred3, axis = 1)
for i in range(0,testRow):
    pred2[i] = pred3[i]*(1/pred3[i,argmaxPred3[i]])
    

thr = 2
pth = rootPath +str(thr) + '_' + '.csv'
f = open(pth, 'w')
f.write('"id","tags"\n')
for i in range(0,testRow):
    lb = '"'
    for j in range(0,38):       
        if (pred[i,j]+pred1[i,j]+ pred2[i,j]+pred3[i,j]) >= thr:
            lb += (listLabel[j] + ' ')
    lb = lb[:len(lb)-1]
    lb += '"\n'
    f.write('"' + str(i) + '"'+','+ lb)
f.close()

thr = 2.4
pth = rootPath +str(thr) + '_' + '.csv'
f = open(pth, 'w')
f.write('"id","tags"\n')
for i in range(0,testRow):
    lb = '"'
    for j in range(0,38):       
        if (pred[i,j]+pred1[i,j]+ pred2[i,j]+pred3[i,j]) >= thr:
            lb += (listLabel[j] + ' ')
    lb = lb[:len(lb)-1]
    lb += '"\n'
    f.write('"' + str(i) + '"'+','+ lb)
f.close()

thr = 1.6
pth = rootPath +str(thr) + '_' + '.csv'
f = open(pth, 'w')
f.write('"id","tags"\n')
for i in range(0,testRow):
    lb = '"'
    for j in range(0,38):       
        if (pred[i,j]+pred1[i,j]+ pred2[i,j]+pred3[i,j]) >= thr:
            lb += (listLabel[j] + ' ')
    lb = lb[:len(lb)-1]
    lb += '"\n'
    f.write('"' + str(i) + '"'+','+ lb)
f.close()

print(acc)
print(acc1)
print(acc2)
print(acc3)

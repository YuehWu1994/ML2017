#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sys
import csv
import keras
import keras.backend as K
import pickle
from numpy import genfromtxt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers import  Dropout, Activation, TimeDistributed, Dense, Input
from keras.layers.recurrent import GRU, LSTM
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint

def f1_score(y_true,y_pred):
    thresh = 0.4
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true * y_pred,axis=-1)
    
    precision=tp/(K.sum(y_pred,axis=-1)+K.epsilon())
    recall=tp/(K.sum(y_true,axis=-1)+K.epsilon())
    return K.mean(2*((precision*recall)/(precision+recall+K.epsilon())))

def unique_list(l):
    ulist = []
    [ulist.append(x) for x in l if x not in ulist]
    return ulist


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

def f1Count(thr, numModel, predTotal, valLabel):
    f1 = np.zeros((497,1))       

       
    thr = numModel*thr
    for i in range(0,497):
        if np.size(np.where(predTotal[i,:] >= thr)) == 0:
            for j in range(0,38):
                if j == np.argmax(predTotal[i,:]):
                    predTotal[i,j] = 1
                else:
                    predTotal[i,j] = 0
        else:
            for j in range(0,38):
                if (predTotal[i,j]) >= thr:
                    predTotal[i,j] = 1
                else:
                    predTotal[i,j] = 0
        """
        for j in range(0,38):
            if (predTotal[i,j]) >= thr:
                predTotal[i,j] = 1
            else:
                predTotal[i,j] = 0
        """
        count = 0
        for j in range(0,38):
            if predTotal[i,j] == 1 and valLabel[i,j] == 1:
                count += 1
        numPred = float(np.size(np.where(predTotal[i,:] == 1)))
        numTrue = float(np.size(np.where(valLabel[i,:] == 1)))
        #print(count)
        precise = count/numPred
        recal = count/numTrue
        if (precise+recal) == 0:
            f1[i,0] = 0
        else:
            f1[i,0] = 2*precise*recal/(precise+recal)
        #print(f1[i,0])
    print(np.mean(f1))



rootPath = '/home/pc193/093/093Hw5/'
loadPath = '/home/pc193/093/savedModel/'
arg = True
#file = h5py.File(rootPath+"best.hdf5", 'r') 

m0 = str(sys.argv[1])
m1 = str(sys.argv[2])
m2 = str(sys.argv[3])
m3 = str(sys.argv[4])
writeFile = int(sys.argv[5])
model = load_model(loadPath + m0,custom_objects={'fmeasure': fmeasure})
model1 = load_model(loadPath + m1,custom_objects={'fmeasure': fmeasure})
model2 = load_model(loadPath + m2,custom_objects={'f1_score': f1_score})
model3 = load_model(loadPath + m3,custom_objects={'f1_score': f1_score})

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
pickle.dump( sequences, open( loadPath +"save.p", "wb" ),protocol=2)
print(sequences[0])
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
pickle.dump( listLabel, open( loadPath +"label.p", "wb" ),protocol=2)

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
"""
# glove
embeddings_index = {}
f = open(rootPath + 'glove.6B.'+ str(EMBEDDING_DIM) + 'd.txt', 'r')
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
if writeFile == 0:
    thr = 0.5 
    pred = model.predict(valDb)
    argmaxPred = np.argmax(pred, axis = 1)
    for i in range(0,497):
        pred[i] = pred[i]*(1/pred[i,argmaxPred[i]])
        
    pred1 = model1.predict(valDb)
    argmaxPred1 = np.argmax(pred1, axis = 1)
    for i in range(0,497):
        pred1[i] = pred1[i]*(1/pred1[i,argmaxPred1[i]])
        
    pred2 = model2.predict(valDb)
    argmaxPred2 = np.argmax(pred2, axis = 1)
    for i in range(0,497):
        pred2[i] = pred2[i]*(1/pred2[i,argmaxPred2[i]])
        
    pred3 = model3.predict(valDb)
    argmaxPred3 = np.argmax(pred3, axis = 1)
    for i in range(0,497):
        pred3[i] = pred3[i]*(1/pred3[i,argmaxPred3[i]])
         
    predTotal = pred+pred1+pred2+pred3
    predTotal1 = pred+pred1+pred2
    predTotal2 = pred1+pred2+pred3
    predTotal3 = pred+pred1+pred3
    predTotal4 = pred+pred2+pred3
    predTotal5 = pred+pred1
    predTotal6 = pred2+pred3
    predTotal7 = pred1+pred2
    predTotal8 = pred+pred3
    predTotal9 = pred+pred2
    predTotal10 = pred1+pred3
    predTotal11 = pred
    predTotal12 = pred3
    predTotal13 = pred1
    predTotal14 = pred2
    
    print('___' + str(thr) + '_____')
    #numModel = 4
    f1Count(thr, 4,predTotal,valLabel)
    f1Count(thr, 3,predTotal1,valLabel)
    f1Count(thr, 3,predTotal2,valLabel)
    f1Count(thr, 3,predTotal3,valLabel)
    f1Count(thr, 3,predTotal4,valLabel)
    f1Count(thr, 2,predTotal5,valLabel)
    f1Count(thr, 2,predTotal6,valLabel)
    f1Count(thr, 2,predTotal7,valLabel)
    f1Count(thr, 2,predTotal8,valLabel)
    f1Count(thr, 2,predTotal9,valLabel)
    f1Count(thr, 2,predTotal10,valLabel)
    f1Count(thr, 1,predTotal11,valLabel)
    f1Count(thr, 1,predTotal12,valLabel)
    f1Count(thr, 1,predTotal13,valLabel)
    f1Count(thr, 1,predTotal14,valLabel)
    
    
    
    
    thr = 0.45
    pred = model.predict(valDb)
    argmaxPred = np.argmax(pred, axis = 1)
    for i in range(0,497):
        pred[i] = pred[i]*(1/pred[i,argmaxPred[i]])
        
    pred1 = model1.predict(valDb)
    argmaxPred1 = np.argmax(pred1, axis = 1)
    for i in range(0,497):
        pred1[i] = pred1[i]*(1/pred1[i,argmaxPred1[i]])
        
    pred2 = model2.predict(valDb)
    argmaxPred2 = np.argmax(pred2, axis = 1)
    for i in range(0,497):
        pred2[i] = pred2[i]*(1/pred2[i,argmaxPred2[i]])
        
    pred3 = model3.predict(valDb)
    argmaxPred3 = np.argmax(pred3, axis = 1)
    for i in range(0,497):
        pred3[i] = pred3[i]*(1/pred3[i,argmaxPred3[i]])
          
    predTotal = pred+pred1+pred2+pred3
    predTotal1 = pred+pred1+pred2
    predTotal2 = pred1+pred2+pred3
    predTotal3 = pred+pred1+pred3
    predTotal4 = pred+pred2+pred3
    predTotal5 = pred+pred1
    predTotal6 = pred2+pred3
    predTotal7 = pred1+pred2
    predTotal8 = pred+pred3
    predTotal9 = pred+pred2
    predTotal10 = pred1+pred3
    predTotal11 = pred
    predTotal12 = pred3
    predTotal13 = pred1
    predTotal14 = pred2
    
    print('___' + str(thr) + '_____')
    #numModel = 4
    f1Count(thr, 4,predTotal,valLabel)
    f1Count(thr, 3,predTotal1,valLabel)
    f1Count(thr, 3,predTotal2,valLabel)
    f1Count(thr, 3,predTotal3,valLabel)
    f1Count(thr, 3,predTotal4,valLabel)
    f1Count(thr, 2,predTotal5,valLabel)
    f1Count(thr, 2,predTotal6,valLabel)
    f1Count(thr, 2,predTotal7,valLabel)
    f1Count(thr, 2,predTotal8,valLabel)
    f1Count(thr, 2,predTotal9,valLabel)
    f1Count(thr, 2,predTotal10,valLabel)
    f1Count(thr, 1,predTotal11,valLabel)
    f1Count(thr, 1,predTotal12,valLabel)
    f1Count(thr, 1,predTotal13,valLabel)
    f1Count(thr, 1,predTotal14,valLabel)
    
    
    
    thr = 0.4
    pred = model.predict(valDb)
    argmaxPred = np.argmax(pred, axis = 1)
    for i in range(0,497):
        pred[i] = pred[i]*(1/pred[i,argmaxPred[i]])
        
    pred1 = model1.predict(valDb)
    argmaxPred1 = np.argmax(pred1, axis = 1)
    for i in range(0,497):
        pred1[i] = pred1[i]*(1/pred1[i,argmaxPred1[i]])
        
    pred2 = model2.predict(valDb)
    argmaxPred2 = np.argmax(pred2, axis = 1)
    for i in range(0,497):
        pred2[i] = pred2[i]*(1/pred2[i,argmaxPred2[i]])
        
    pred3 = model3.predict(valDb)
    argmaxPred3 = np.argmax(pred3, axis = 1)
    for i in range(0,497):
        pred3[i] = pred3[i]*(1/pred3[i,argmaxPred3[i]])
           
    predTotal = pred+pred1+pred2+pred3
    predTotal1 = pred+pred1+pred2
    predTotal2 = pred1+pred2+pred3
    predTotal3 = pred+pred1+pred3
    predTotal4 = pred+pred2+pred3
    predTotal5 = pred+pred1
    predTotal6 = pred2+pred3
    predTotal7 = pred1+pred2
    predTotal8 = pred+pred3
    predTotal9 = pred+pred2
    predTotal10 = pred1+pred3
    predTotal11 = pred
    predTotal12 = pred3
    predTotal13 = pred1
    predTotal14 = pred2
    
    print('___' + str(thr) + '_____')
    #numModel = 4
    f1Count(thr, 4,predTotal,valLabel)
    f1Count(thr, 3,predTotal1,valLabel)
    f1Count(thr, 3,predTotal2,valLabel)
    f1Count(thr, 3,predTotal3,valLabel)
    f1Count(thr, 3,predTotal4,valLabel)
    f1Count(thr, 2,predTotal5,valLabel)
    f1Count(thr, 2,predTotal6,valLabel)
    f1Count(thr, 2,predTotal7,valLabel)
    f1Count(thr, 2,predTotal8,valLabel)
    f1Count(thr, 2,predTotal9,valLabel)
    f1Count(thr, 2,predTotal10,valLabel)
    f1Count(thr, 1,predTotal11,valLabel)
    f1Count(thr, 1,predTotal12,valLabel)
    f1Count(thr, 1,predTotal13,valLabel)
    f1Count(thr, 1,predTotal14,valLabel)
    
    
    thr = 0.55
    pred = model.predict(valDb)
    argmaxPred = np.argmax(pred, axis = 1)
    for i in range(0,497):
        pred[i] = pred[i]*(1/pred[i,argmaxPred[i]])
        
    pred1 = model1.predict(valDb)
    argmaxPred1 = np.argmax(pred1, axis = 1)
    for i in range(0,497):
        pred1[i] = pred1[i]*(1/pred1[i,argmaxPred1[i]])
        
    pred2 = model2.predict(valDb)
    argmaxPred2 = np.argmax(pred2, axis = 1)
    for i in range(0,497):
        pred2[i] = pred2[i]*(1/pred2[i,argmaxPred2[i]])
        
    pred3 = model3.predict(valDb)
    argmaxPred3 = np.argmax(pred3, axis = 1)
    for i in range(0,497):
        pred3[i] = pred3[i]*(1/pred3[i,argmaxPred3[i]])
           
    predTotal = pred+pred1+pred2+pred3
    predTotal1 = pred+pred1+pred2
    predTotal2 = pred1+pred2+pred3
    predTotal3 = pred+pred1+pred3
    predTotal4 = pred+pred2+pred3
    predTotal5 = pred+pred1
    predTotal6 = pred2+pred3
    predTotal7 = pred1+pred2
    predTotal8 = pred+pred3
    predTotal9 = pred+pred2
    predTotal10 = pred1+pred3
    predTotal11 = pred
    predTotal12 = pred3
    predTotal13 = pred1
    predTotal14 = pred2
    
    print('___' + str(thr) + '_____')
    #numModel = 4
    f1Count(thr, 4,predTotal,valLabel)
    f1Count(thr, 3,predTotal1,valLabel)
    f1Count(thr, 3,predTotal2,valLabel)
    f1Count(thr, 3,predTotal3,valLabel)
    f1Count(thr, 3,predTotal4,valLabel)
    f1Count(thr, 2,predTotal5,valLabel)
    f1Count(thr, 2,predTotal6,valLabel)
    f1Count(thr, 2,predTotal7,valLabel)
    f1Count(thr, 2,predTotal8,valLabel)
    f1Count(thr, 2,predTotal9,valLabel)
    f1Count(thr, 2,predTotal10,valLabel)
    f1Count(thr, 1,predTotal11,valLabel)
    f1Count(thr, 1,predTotal12,valLabel)
    f1Count(thr, 1,predTotal13,valLabel)
    f1Count(thr, 1,predTotal14,valLabel)
    
    
    
    thr = 0.6
    pred = model.predict(valDb)
    argmaxPred = np.argmax(pred, axis = 1)
    for i in range(0,497):
        pred[i] = pred[i]*(1/pred[i,argmaxPred[i]])
        
    pred1 = model1.predict(valDb)
    argmaxPred1 = np.argmax(pred1, axis = 1)
    for i in range(0,497):
        pred1[i] = pred1[i]*(1/pred1[i,argmaxPred1[i]])
        
    pred2 = model2.predict(valDb)
    argmaxPred2 = np.argmax(pred2, axis = 1)
    for i in range(0,497):
        pred2[i] = pred2[i]*(1/pred2[i,argmaxPred2[i]])
        
    pred3 = model3.predict(valDb)
    argmaxPred3 = np.argmax(pred3, axis = 1)
    for i in range(0,497):
        pred3[i] = pred3[i]*(1/pred3[i,argmaxPred3[i]])
           
    predTotal = pred+pred1+pred2+pred3
    predTotal1 = pred+pred1+pred2
    predTotal2 = pred1+pred2+pred3
    predTotal3 = pred+pred1+pred3
    predTotal4 = pred+pred2+pred3
    predTotal5 = pred+pred1
    predTotal6 = pred2+pred3
    predTotal7 = pred1+pred2
    predTotal8 = pred+pred3
    predTotal9 = pred+pred2
    predTotal10 = pred1+pred3
    predTotal11 = pred
    predTotal12 = pred3
    predTotal13 = pred1
    predTotal14 = pred2
    
    print('___' + str(thr) + '_____')
    #numModel = 4
    f1Count(thr, 4,predTotal,valLabel)
    f1Count(thr, 3,predTotal1,valLabel)
    f1Count(thr, 3,predTotal2,valLabel)
    f1Count(thr, 3,predTotal3,valLabel)
    f1Count(thr, 3,predTotal4,valLabel)
    f1Count(thr, 2,predTotal5,valLabel)
    f1Count(thr, 2,predTotal6,valLabel)
    f1Count(thr, 2,predTotal7,valLabel)
    f1Count(thr, 2,predTotal8,valLabel)
    f1Count(thr, 2,predTotal9,valLabel)
    f1Count(thr, 2,predTotal10,valLabel)
    f1Count(thr, 1,predTotal11,valLabel)
    f1Count(thr, 1,predTotal12,valLabel)
    f1Count(thr, 1,predTotal13,valLabel)
    f1Count(thr, 1,predTotal14,valLabel)
    #a = fmeasure(valLabel,pred)
    #print(a)
else:
    print("writeFile...")
    thr = float(sys.argv[6])
    numModel = int(sys.argv[7])
    idx = int(sys.argv[8])
    
    #print(testData[708,0:38])
    
    pred = model.predict(testData)
    argmaxPred = np.argmax(pred, axis = 1)
    for i in range(0,testRow):
        pred[i] = pred[i]*(1/pred[i,argmaxPred[i]])
    print(pred[0])
        
    pred1 = model1.predict(testData)
    argmaxPred1 = np.argmax(pred1, axis = 1)
    for i in range(0,testRow):
        pred1[i] = pred1[i]*(1/pred1[i,argmaxPred1[i]])
        
    pred2 = model2.predict(testData)
    argmaxPred2 = np.argmax(pred2, axis = 1)
    for i in range(0,testRow):
        pred2[i] = pred2[i]*(1/pred2[i,argmaxPred2[i]])
        
    pred3 = model3.predict(testData)
    argmaxPred3 = np.argmax(pred3, axis = 1)
    for i in range(0,testRow):
        pred3[i] = pred3[i]*(1/pred3[i,argmaxPred3[i]])
         
    
    predTotal = pred+pred1+pred2+pred3
    predTotal1 = pred+pred1+pred2
    predTotal2 = pred1+pred2+pred3
    predTotal3 = pred+pred1+pred3
    predTotal4 = pred+pred2+pred3
    predTotal5 = pred+pred1
    predTotal6 = pred2+pred3
    predTotal7 = pred1+pred2
    predTotal8 = pred+pred3
    predTotal9 = pred+pred2
    predTotal10 = pred1+pred3
    predTotal11 = pred
    predTotal12 = pred3
    predTotal13 = pred1
    predTotal14 = pred2    
    
    if idx == 0:
        pT = predTotal
    elif idx == 1:
        pT = predTotal1
    elif idx == 2:
        pT = predTotal2
    elif idx == 3:
        pT = predTotal3
    elif idx == 4:
        pT = predTotal4
    elif idx == 5:
        pT = predTotal5
    elif idx == 6:
        pT = predTotal6
    elif idx == 7:
        pT = predTotal7
    elif idx == 8:
        pT = predTotal8
    elif idx == 9:
        pT = predTotal9
    elif idx == 10:
        pT = predTotal10
    elif idx == 11:
        pT = predTotal11
    elif idx == 12:
        pT = predTotal12
    elif idx == 13:
        pT = predTotal13
    else:
        pT = predTotal14
                
    pth = rootPath +str(thr) + '_' + '.csv'
    f = open(pth, 'w')
    f.write('"id","tags"\n')
    for i in range(0,testRow):
        lb = '"'
        if np.size(np.where(pT[i,:] >= (thr*numModel))) == 0:
            print(thr)
            for j in range(0,38):
                if j == np.argmax(pT[i,:]):
                    lb += (listLabel[j] + ' ')
        else:
            for j in range(0,38):       
                if (pT[i,j]) >= (thr*numModel):
                    lb += (listLabel[j] + ' ')
        lb = lb[:len(lb)-1]
        lb += '"\n'
        f.write('"' + str(i) + '"'+','+ lb)
    f.close()

"""

    pth = rootPath +str(thr) + '_' + '.csv'
    f = open(pth, 'w')
    f.write('"id","tags"\n')
    for i in range(0,testRow):
        lb = '"'
        if np.size(np.where(predTotal1[i,:] >= (thr*3))) == 0:
            for j in range(0,38):
                if j == np.argmax(predTotal1[i,:]):
                    lb += (listLabel[j] + ' ')
        else:
            for j in range(0,38):       
                if (predTotal1[i,j]) >= (thr*3):
                    lb += (listLabel[j] + ' ')
            lb = lb[:len(lb)-1]
            lb += '"\n'
            f.write('"' + str(i) + '"'+','+ lb)
    f.close()
    
    
    pth = rootPath +str(thr) + '_' + '.csv'
    f = open(pth, 'w')
    f.write('"id","tags"\n')
    for i in range(0,testRow):
        lb = '"'
        for j in range(0,38):       
            if (predTotal1[i,j]) >= (thr*3):
                lb += (listLabel[j] + ' ')
        lb = lb[:len(lb)-1]
        lb += '"\n'
        f.write('"' + str(i) + '"'+','+ lb)
    f.close()
    
    
        if np.size(np.where(predTotal[i,:] >= thr)) == 0:
            for j in range(0,38):
                if j == np.argmax(predTotal[i,:]):
                    predTotal[i,j] = 1
                else:
                    predTotal[i,j] = 0
        else:
            for j in range(0,38):
                if (predTotal[i,j]) >= thr:
                    predTotal[i,j] = 1
                else:
                    predTotal[i,j] = 0
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np
import sys
import keras.backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import pickle

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
    
m0 = '2.h5'
m1 = '1.h5'
m2 = '3.h5'
m3 = '4.h5'

model3 = load_model(m3,custom_objects={'f1_score': f1_score})
model2 = load_model(m2,custom_objects={'f1_score': f1_score})
model1 = load_model(m1,custom_objects={'fmeasure': fmeasure})
model = load_model(m0,custom_objects={'fmeasure': fmeasure})



rootPath = '/home/pc193/093/093Hw5/'
arg = True
setVal = True
MAX_SEQUENCE_LENGTH = 320
testLine = []

row = 4964
valId = int(row * 0.9)
# testInit
if arg == True:
    testPath = sys.argv[1]
else:
    testPath = rootPath + 'test_data.csv'
with open(testPath, 'r') as df:
    for l in df:
        spl = l.split(',') 
        testLine.append(','.join(spl[1:]))    
testLine = testLine[1:]
testRow = len(testLine)

listLabel = pickle.load( open("label.p", "rb" ))
sequences = pickle.load( open( "save.p", "rb" ))
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
testData  = data[row:]


thr = 0.3925
numModel = 4
idx = 0

pred3 = model3.predict(testData)
argmaxPred3 = np.argmax(pred3, axis = 1)
for i in range(0,testRow):
    pred3[i] = pred3[i]*(1/pred3[i,argmaxPred3[i]])
    
pred2 = model2.predict(testData)
argmaxPred2 = np.argmax(pred2, axis = 1)
for i in range(0,testRow):
    pred2[i] = pred2[i]*(1/pred2[i,argmaxPred2[i]])

pred1 = model1.predict(testData)
argmaxPred1 = np.argmax(pred1, axis = 1)
for i in range(0,testRow):
    pred1[i] = pred1[i]*(1/pred1[i,argmaxPred1[i]])  

pred = model.predict(testData)
argmaxPred = np.argmax(pred, axis = 1)
for i in range(0,testRow):
    pred[i] = pred[i]*(1/pred[i,argmaxPred[i]])


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
    
if arg == True:
    pth = sys.argv[2]
else:
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




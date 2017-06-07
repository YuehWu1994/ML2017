#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sys
import csv
import keras
import keras.backend as K
import keras.models as kmodels
import keras.layers as klayers
from numpy import genfromtxt
from keras.callbacks import EarlyStopping, ModelCheckpoint

debugArg = True
if debugArg == True:
    batchSz = int(sys.argv[1])
    epoch = int(sys.argv[2])
    patiences = int(sys.argv[3])
    opt = str(sys.argv[4])
    denseAct = str(sys.argv[5])
    outAct = str(sys.argv[6])    
    hidLayer = int(sys.argv[7])
    lay1 = int(sys.argv[8])
    lay2 = int(sys.argv[9])
    lay3 = int(sys.argv[10])
    dDp1 = float(sys.argv[11])
    dDp2 = float(sys.argv[12])
    dDp3 = float(sys.argv[13])
    valUse = int(sys.argv[14])
    embDim = int(sys.argv[15])


else:
    batchSz = 128
    epoch = 100
    patiences = 70
    opt = 'adam'
    denseAct = 'relu'
    outAct = 'linear'
    hidLayer = 3
    lay1 = 128
    lay2 = 128
    lay3 = 128
    dDp1 = 0.5
    dDp2 = 0.5
    dDp3 = 0.5
    valUse = 0
    embDim = 64
    
log = (str(batchSz) + '_' + str(epoch) + '_' + str(patiences) + '_'  
      + str(opt)+ '_'  + str(denseAct) + '_'  + str(outAct) + '_' + str(hidLayer)
      + '_'  + str(lay1) + '_'  + str(lay2) + '_'  + str(lay3) + '_'
      + str(dDp1)+ '_'  + str(dDp2) + '_'  + str(dDp3) + '_' + str(valUse))+ '_' + str(embDim)

arg = False
#rootPath = '/Users/apple/desktop/NTUEE/課程八/ML/git/hw6/'
#rootPath = '/home/bl418/桌面/093/hw6/'
rootPath = '/home/pc193/093/093Hw6/'
train_file = 'train.csv'
user_file = 'users.csv'
movie_file = 'movies.csv'

users_detail = []
users = []
movies_detail = []
movies = []
n_movies = 3952
n_users = 6040
valSplit = 0.92
RNG_SEED = 1446557


if arg == True:
    df = pd.read_csv(sys.argv[1],encoding="big5")
else:   
    df = pd.read_csv('train.csv',encoding="big5")
df = df.sample(frac=1., random_state=RNG_SEED)
rawData = df.as_matrix()
row = np.size(rawData[:,0])
rating = rawData[:,3]
user = rawData[:,1]
movie = rawData[:,2]

if valUse == 1:
    splitTo = int(valSplit*row)
    trainUserX = user[:splitTo]
    trainMovieX = movie[:splitTo]
    trainY = rating[:splitTo]
    valUserX = user[splitTo:]
    valMovieX = movie[splitTo:]
    valY = rating[splitTo:]
else:
    trainUserX = user
    trainMovieX = movie
    trainY = rating

del user, rating, df
# , encoding = 'ISO-8859-1'
with open(user_file, 'r') as df:
    for l in df:
        spl = l.split('::')
        users_detail.append(spl)   
        users.append(spl[0])
    users_detail = users_detail[1:]
    users = users[1:]
with open(movie_file, 'r') as df:
    for l in df:
        spl = l.split('::')
        movies_detail.append(spl)   
        movies.append(spl[0])
    movies_detail = movies_detail[1:]
    movies = movies[1:]
    
movie_input = keras.layers.Input(shape=[1])
movie_vec = keras.layers.Flatten()(keras.layers.Embedding(n_movies + 1, embDim)(movie_input))
#movie_vec = keras.layers.Dropout(0.5)(movie_vec)

user_input = keras.layers.Input(shape=[1])
user_vec = keras.layers.Flatten()(keras.layers.Embedding(n_users + 1, embDim)(user_input))
#user_vec = keras.layers.Dropout(0.5)(user_vec)

input_vecs = keras.layers.merge([movie_vec, user_vec], mode='concat')
#input_vecs = keras.layers.merge([movie_vec, user_vec], mode='dot', dot_axes=1)
if hidLayer > 0:
    nn = keras.layers.Dropout(dDp1)(keras.layers.Dense(lay1, activation=denseAct)(input_vecs))
else:
    nn = input_vecs

if hidLayer > 1:
    #nn = keras.layers.normalization.BatchNormalization()(nn)
    nn = keras.layers.Dropout(dDp2)(keras.layers.Dense(lay2, activation=denseAct)(nn))
if hidLayer > 2:  
    #nn = keras.layers.normalization.BatchNormalization()(nn)
    nn = keras.layers.Dense(lay3, activation=denseAct)(nn)

result = keras.layers.Dense(1, activation=outAct)(nn)

model = kmodels.Model([movie_input, user_input], result)
model.compile(opt, 'mean_squared_error', metrics=['accuracy'])
model.summary()

if valUse == 1:
    earlystopping = EarlyStopping(monitor='val_loss', patience = patiences, verbose=1, mode='min')
    checkpoint = ModelCheckpoint(filepath= log + '_' + ".hdf5",
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=False,
                                     monitor='val_loss',
                                     mode='min')
    
    h = model.fit([trainMovieX,trainUserX], trainY,batch_size=batchSz,epochs=epoch,validation_data=([valMovieX,valUserX] ,valY),callbacks=[earlystopping,checkpoint])
else:
    earlystopping = EarlyStopping(monitor='loss', patience = patiences, verbose=1, mode='min')
    checkpoint = ModelCheckpoint(filepath= log + '_' + ".hdf5",
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=False,
                                     monitor='loss',
                                     mode='min')
    
    h = model.fit([trainMovieX,trainUserX], trainY,batch_size=batchSz,epochs=epoch,callbacks=[earlystopping,checkpoint])    


#score, acc = model.evaluate([valUserX, valMovieX] ,valY,batch_size=batchSz)
#print('Test score:', score)
#print('Test accuracy:', acc)
#print('model min loss: ', min(h.history['val_loss']))

if valUse == 1:
    with open (str(min(h.history['val_loss'])) + '_' +log + '.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow('a')
else:
    with open (str(min(h.history['loss'])) + '_' +log + '.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow('a')
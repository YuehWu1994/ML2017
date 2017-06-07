import numpy as np
import pandas as pd
import sys
import keras.models as kmodels
import keras.layers as klayers
import keras.backend as K
import keras
from keras.layers import Embedding, Reshape, Merge, Dropout, Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.utils.vis_utils import plot_model
import pickle

arg = True

def mf_model(num_users, num_movies, latent_dim):
    user_input = keras.layers.Input(shape=[1])
    movie_input = keras.layers.Input(shape=[1])
    
    user_vec = Embedding(num_users, latent_dim, embeddings_initializer='random_normal')(user_input)
    user_vec = keras.layers.Flatten()(user_vec)
    movie_vec = Embedding(num_movies, latent_dim, embeddings_initializer='random_normal')(movie_input)
    movie_vec = keras.layers.Flatten()(movie_vec)
    user_bias = Embedding(num_users, 1, embeddings_initializer='zero')(user_input)
    user_bias = keras.layers.Flatten()(user_bias)
    movie_bias = Embedding(num_movies, 1, embeddings_initializer='zero')(movie_input)
    movie_bias = keras.layers.Flatten()(movie_bias)
    r_hat = keras.layers.Dot(axes=1)([movie_vec, user_vec])
    r_hat = keras.layers.Add()([r_hat, movie_bias, user_bias])
    model = keras.models.Model([movie_input, user_input], r_hat)
    model.compile(loss = 'mse', optimizer = 'adam')
	
    return model

# Matrix Factorization
model = mf_model(6041, 3953, 40)
# load weights
model.load_weights('0.87504.hdf5')


if arg == True:
    df = pd.read_csv(sys.argv[1]+'test.csv',encoding="big5")
else:   
    df = pd.read_csv('test.csv',encoding="big5")
test = df.as_matrix()
userid_test = test[:,1].astype('int')
movieid_test = test[:,2].astype('int')
#mean = pickle.load( open("mean.p", "rb" ))
#std = pickle.load( open( "std.p", "rb" ))
mean = 3.58171208604
std =  1.11689766115
# Predict
pred = model.predict([movieid_test, userid_test])*std + mean

# Output result
pth = sys.argv[2]
file_write = open(pth, 'w')
file_write.write('TestDataID,Rating\n')
for i,j in enumerate(pred):
	file_write.write(str(i+1) + ',' + str(float(j)) + '\n')


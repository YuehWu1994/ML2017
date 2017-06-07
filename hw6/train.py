
# coding: utf-8

# In[ ]:

import math
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
import numpy as np
from itertools import product

from keras import backend as K
from keras.layers import Input, Embedding, Flatten, Lambda, Dense, BatchNormalization, Dropout, LSTM
from keras.layers.merge import add, dot, concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.regularizers import l2
from keras.utils import plot_model



def MF(n_U,n_M,F):    
    U_input = Input(shape=(1,), dtype='int64', name='users')
    M_input = Input(shape=(1,), dtype='int64', name='movies')

    U_embedding = Embedding(input_dim=max_userid, output_dim=F)(U_input)
    M_embedding = Embedding(input_dim=max_movieid, output_dim=F)(M_input)

    predicted_preference = dot(inputs=[U_embedding, M_embedding], axes=2)
    predicted_preference = Flatten()(predicted_preference)
    
    model = Model(inputs=[U_input, M_input],outputs=predicted_preference)
    return model


#Define constants
RATINGS_CSV_FILE = 'train.csv'
MODEL_WEIGHTS_FILE = 'norMF120.h5'
MODEL_PLOT = 'network_norMF120.png'
TRAINING_PROCESS = 'norMF120.png'
VISUALIZE = 'norMF_vis120.png'
K_FACTORS = 120
RNG_SEED = 1446557



#Load data
ratings = pd.read_csv(RATINGS_CSV_FILE,  
                      usecols=['TrainDataID','UserID', 'MovieID', 'Rating'])
max_userid = ratings['UserID'].drop_duplicates().max()
max_movieid = ratings['MovieID'].drop_duplicates().max()
print (len(ratings), 'ratings loaded.')



#Create training set
shuffled_ratings = ratings.sample(frac=1., random_state=RNG_SEED)
Users = shuffled_ratings['UserID'].values
print ('Users:', Users, ', shape =', Users.shape)
Movies = shuffled_ratings['MovieID'].values
print ('Movies:', Movies, ', shape =', Movies.shape)
Ratings = shuffled_ratings['Rating'].values
print ('Ratings:', Ratings, ', shape =', Ratings.shape)


# Ceate model
model = MF(max_userid, max_movieid,K_FACTORS)
model.compile(loss='mse', optimizer='adamax')
#plot_model(model, to_file=MODEL_PLOT)






#Train model
callbacks = [EarlyStopping('val_loss', patience=2), 
             ModelCheckpoint(MODEL_WEIGHTS_FILE, save_best_only=True)]
history = model.fit([Users, Movies], Ratings, nb_epoch=100, validation_split=.1, verbose=1, callbacks=callbacks)





#Print best validation RMSE
min_val_loss, idx = min((val, idx) for (idx, val) in enumerate(history.history['val_loss']))
print ('Minimum RMSE at epoch', '{:d}'.format(idx+1), '=', '{:.4f}'.format(math.sqrt(min_val_loss)))



#Predict
model.load_weights(MODEL_WEIGHTS_FILE)
TESTING_CSV_FILE = 'test.csv'
test = pd.read_csv(TESTING_CSV_FILE, usecols=['TestDataID', 'UserID', 'MovieID'])
print (len(test), 'descriptions of', max_movieid, 'movies loaded.')

prediction = model.predict([test['UserID'].values,test['MovieID'].values])
ids = test['TestDataID'].values

"""
with open('submit.csv','w') as sm:
    print('\"TestDataID\",\"Rating\"',file=sm)
    for i in range(len(ids )):
        print('\"%d\",\"%.1f\"'%(ids[i],prediction[i]),file=sm)
"""
f = open("submit.csv", 'w')
f.write('TestDataID,Rating\n')

for i in range(0,len(ids)):
    f.write(str(ids[i]) + ',' + str(float(prediction[i])) + '\n')
f.close()  





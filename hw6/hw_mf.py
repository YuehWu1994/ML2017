import numpy as np
import pandas as pd
import sys
from sklearn import dummy, metrics, cross_validation, ensemble
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

user_file = 'users.csv'
movie_file = 'movies.csv'


movies_detail = []
movies = []
movies_cate = []
movies_id = []

argParam = True
if argParam == True:
    normalize = int(sys.argv[1])
    lat_dim = int(sys.argv[2])
    biasOrNot = int(sys.argv[3])
else:
    normalize = 1
    lat_dim = 20
    biasOrNot = 1   


log = str(normalize)+ '_' +str(lat_dim)+ '_' +str(biasOrNot)

def mf_model(num_users, num_movies, num_cat, latent_dim):
    user_input = keras.layers.Input(shape=[1])
    movie_input = keras.layers.Input(shape=[1])
    
    cat_input = keras.layers.Input(shape=[1])
    
    user_vec = Embedding(num_users, latent_dim, embeddings_initializer='random_normal')(user_input)
    user_vec = keras.layers.Flatten()(user_vec)
    movie_vec = Embedding(num_movies, latent_dim, embeddings_initializer='random_normal')(movie_input)
    movie_vec = keras.layers.Flatten()(movie_vec)
    """
    cat_vec = Embedding(num_cat, 6, embeddings_initializer='random_normal')(cat_input)
    cat_vec = keras.layers.Flatten()(cat_vec)
    """
    if biasOrNot == 1:
        user_bias = Embedding(num_users, 1, embeddings_initializer='zero')(user_input)
        user_bias = keras.layers.Flatten()(user_bias)
        movie_bias = Embedding(num_movies, 1, embeddings_initializer='zero')(movie_input)
        movie_bias = keras.layers.Flatten()(movie_bias)
        
        cat_bias = Embedding(num_cat, 1, embeddings_initializer='zero')(cat_input)
        cat_bias = keras.layers.Flatten()(cat_bias)
        
    #r_hat = keras.layers.Dot(axes=1)([movie_vec, user_vec])
    r_hat = keras.layers.Dot(axes=1)([movie_vec, user_vec])
    if biasOrNot == 1:
        #r_hat = keras.layers.Add()([r_hat, movie_bias, user_bias])
        r_hat = keras.layers.Add()([r_hat, movie_bias, user_bias, cat_bias])
    #model = keras.models.Model([movie_input, user_input], r_hat)
    model = keras.models.Model([movie_input, user_input, cat_input], r_hat)
    model.compile(loss = 'mse', optimizer = 'adam')
	
    return model


# Read training data
df = pd.read_csv('train.csv', encoding="big5")
df = df.sample(frac=1., random_state=1446557)
ratings = df.as_matrix()

rating_matrix = ratings[:,2]
movieid = rating_matrix.astype('int')
rating_matrix = ratings[:,1]
userid = rating_matrix.astype('int')
labels = ratings[:,3].astype('int')
del ratings,rating_matrix

mean = np.mean(labels)
pickle.dump( mean, open("mean.p", "wb" ),protocol=2)
std = np.std(labels)
pickle.dump( std, open("std.p", "wb" ),protocol=2)


# Read Movie
with open(movie_file, 'r') as df:
    for l in df:
        spl = l.split('::')
        movies_detail.append(spl)   
        movies.append(spl[0])
    movies_detail = movies_detail[1:]
    for l in movies_detail:
        movies_id.append(int(l[0]))
        c = l[2].split('|')
        if c[0] == 'Animation' or c[0] == "Children's" or c[0] == 'Animation\n' or c[0] == "Children's\n":
            movies_cate.append(0)
        elif c[0] == 'Thriller' or c[0] == 'Horror' or c[0] == 'Crime' or c[0] == 'Thriller\n' or c[0] == 'Horror\n' or c[0] == 'Crime\n':
            movies_cate.append(1)
        elif c[0] == 'Comedy' or c[0] == 'Drama' or c[0] == 'Musical'or c[0] == 'Comedy\n' or c[0] == 'Drama\n' or c[0] == 'Musical\n':
            movies_cate.append(2)
        elif c[0] == 'Action' or c[0] == 'Adventure' or c[0] == 'Action\n' or c[0] == 'Adventure\n':
            movies_cate.append(3)
        else :
            movies_cate.append(4)
        
    movies = movies[1:]
    movies_cate = np.array(movies_cate)
    movies_id = np.array(movies_id)
del movies, movies_detail

# Set category 
row = np.size(labels)
catSet = np.zeros((row))
for i in range(0,row):
    if np.size(np.where(movies_id == movieid[i])[0]) > 0:
        idx = np.where(movies_id == movieid[i])[0][0]
        catSet[i] = movies_cate[idx]
    else:
        catSet[i] = 5
    

# Matrix Factorization
model = mf_model(6041, 3953, 7, lat_dim)

es = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
ck = ModelCheckpoint(filepath=(log+'.hdf5'), verbose=1, save_best_only=True, save_weights_only=True, monitor='val_loss', mode='min')

#a_movieid, b_movieid, a_userid, b_userid, a_labels, b_labels = cross_validation.train_test_split(movieid, userid, labels)
a_movieid, b_movieid, a_userid, b_userid, a_catSet, b_catSet,a_labels, b_labels = cross_validation.train_test_split(movieid, userid, catSet ,labels)
# Normalize labels
if normalize == 1:
    a_labels = (a_labels - mean) / std
    b_labels = (b_labels - mean) / std
#h = model.fit([a_movieid, a_userid], a_labels, nb_epoch=500, batch_size=128, validation_data=([b_movieid, b_userid], b_labels), callbacks=[es, ck])
h = model.fit([a_movieid, a_userid, a_catSet], a_labels, nb_epoch=500, batch_size=128, validation_data=([b_movieid, b_userid, b_catSet], b_labels), callbacks=[es, ck])

# load weights
model.load_weights(log+'.hdf5')

# plot model
plot_model(model,to_file='modelMf.png')

# Evaluate
a_movieid, b_movieid, a_userid, b_userid, a_labels, b_labels = cross_validation.train_test_split(movieid, userid, labels)

train_result = metrics.mean_squared_error(a_labels, model.predict([a_movieid, a_userid, a_catSet])*std + mean)
test_result = metrics.mean_squared_error(b_labels, model.predict([b_movieid, b_userid, b_catSet])*std + mean)
print('Training accuracy:' + str(np.sqrt(train_result)))
print('Testing accuracy:' + str(np.sqrt(test_result)))


user_emb = np.array(model.layers[3].get_weights()).squeeze()
movie_emb = np.array(model.layers[2].get_weights()).squeeze()
np.save('user_emb.npy',user_emb)
np.save('movie_emb.npy',movie_emb)


# Predict
df = pd.read_csv('test.csv',encoding="big5")
test = df.as_matrix()
userid_test = test[:,1].astype('int')
movieid_test = test[:,2].astype('int')
# Set category 
row = np.size(movieid_test)
catSet = np.zeros((row))
for i in range(0,row):
    if np.size(np.where(movies_id == movieid_test[i])[0]) > 0:
        idx = np.where(movies_id == movieid_test[i])[0][0]
        catSet[i] = movies_cate[idx]
    else:
        catSet[i] = 5

pred = model.predict([movieid_test, userid_test, catSet])*std + mean

# Output result
file_write = open(log+'_'+ str(min(h.history['val_loss']))+ '_' +'.csv', 'w')
file_write.write('TestDataID,Rating\n')
for i,j in enumerate(pred):
	file_write.write(str(i+1) + ',' + str(float(j)) + '\n')


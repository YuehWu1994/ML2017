import numpy as np
import pandas as pd
import keras.models as kmodels
import keras.layers as klayers
import keras.backend as K
import keras
from keras.layers import Embedding, Reshape, Merge, Dropout, Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from matplotlib import pyplot as plt
#from tsne import bh_sne
from sklearn.manifold import TSNE

user_file = 'users.csv'
movie_file = 'movies.csv'

users_detail = []
users = []
movies_detail = []
movies = []
movies_cate = []
movies_id = []


df = pd.read_csv('train.csv', encoding="big5")
df = df.sample(frac=1., random_state=1446557)
ratings = df.as_matrix()

rating_matrix = ratings[:,2]
movieid = rating_matrix.astype('int')
rating_matrix = ratings[:,1]
userid = rating_matrix.astype('int')
labels = ratings[:,3].astype('int')

mean = np.mean(labels)
std = np.std(labels)

# Matrix Factorization
# , encoding = 'ISO-8859-1'
with open(user_file, 'r', encoding = 'ISO-8859-1') as df:
    for l in df:
        spl = l.split('::')
        users_detail.append(spl)   
        users.append(spl[0])
    users_detail = users_detail[1:]
    users = users[1:]
with open(movie_file, 'r', encoding = 'ISO-8859-1') as df:
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
            movies_cate.append(5)
        elif c[0] == 'Comedy' or c[0] == 'Drama' or c[0] == 'Musical'or c[0] == 'Comedy\n' or c[0] == 'Drama\n' or c[0] == 'Musical\n':
            movies_cate.append(10)
        else:
            movies_cate.append(15)
        """
        elif c[0] == 'Action' or c[0] == 'Adventure' or c[0] == 'Action\n' or c[0] == 'Adventure\n':
            movies_cate.append(3)
        """
        
    movies = movies[1:]
    movies_cate = np.array(movies_cate)
    movies_id = np.array(movies_id)

row = len(movies_cate)
cat = np.array(movies_cate)

    
emb = np.load('movie_emb.npy')
tModel = TSNE(n_components=2, random_state=0)
vec = tModel.fit_transform(emb) 
vec = vec[movies_id-1]
vis_x = vec[:,0]
vis_y = vec[:,1]
cm = plt.cm.get_cmap('RdYlBu')

sc = plt.scatter(vis_x,vis_y,c=cat,cmap=cm,s=5)
plt.colorbar(sc)
plt.show() 

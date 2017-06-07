#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sys
import keras.backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.utils.vis_utils import plot_model



model = load_model('85926.hdf5')
#plot_model(model,to_file='modelDnn.png')

#rootPath = '/Users/apple/desktop/NTUEE/課程八/ML/git/hw6/'
rootPath = '/home/bl418/hahahah/093/hw6/'
#rootPath = '/home/pc193/093/093Hw6/'

arg = True

if arg == True:
    df = pd.read_csv(sys.argv[1]+'test.csv',encoding="big5")
else:   
    df = pd.read_csv(rootPath + 'test.csv',encoding="big5")
rawData = df.as_matrix()
row = np.size(rawData[:,0])
user = rawData[:,1]
movie = rawData[:,2]

del df
pred = model.predict([movie,user])

pth = sys.argv[2]
#regre = int(sys.argv[3])
f = open(pth, 'w')
f.write('TestDataID,Rating\n')

for i in range(0,row):
    f.write(str(i+1) + ',' + str(pred[i][0]) + '\n')
f.close()    
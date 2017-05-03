#!/usr/bin/env python
# -- coding: utf-8 --

import os
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
import numpy as np
import pandas as pd

root_dir = '/Users/apple/desktop/NTUEE/課程八/ML/git/hw3/' 
base_dir = '/Users/apple/desktop/NTUEE/課程八/ML/git/hw3/reportSimCode/p5'

def main():
    model_name = "0.67177.h5"
    model_path = os.path.join(base_dir, model_name)
    emotion_classifier = load_model(model_path)
    layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[1:])

    input_img = emotion_classifier.input
    name_ls = ["conv2d_1","conv2d_2","conv2d_3","conv2d_4","conv2d_5","conv2d_6"]
    #name_ls = ["dense_2","dense_3"]
    collect_layers = [ K.function([input_img, K.learning_phase()], [layer_dict[name].output]) for name in name_ls ]
    
    df = pd.read_csv(root_dir + 'train.csv',encoding="big5")
    rawData = df.as_matrix()
    dfS = len(df)
    private_pixels = [[] for _ in range(dfS)]
    for i in range(0, dfS):
        private_pixels[i] = np.asarray(list(map(int,rawData[i,1].split(' '))))
    private_pixels = np.asarray(private_pixels).astype(float)/255
    private_pixels = private_pixels.reshape(28709,48,48,1)
    del df, rawData
    
    choose_id = 33
    photo = np.reshape(private_pixels[choose_id],(1,48,48,1))
    for cnt, fn in enumerate(collect_layers):
        im = fn([photo, 0]) #get the output of that layer
        fig = plt.figure(figsize=(14, 8))
        nb_filter = int(im[0].shape[3]/16)*16
        
        for i in range(nb_filter-1):
            ax = fig.add_subplot(nb_filter/16, 16, i+1)
            ax.imshow(im[0][0, :, :, i], cmap='BuGn')
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            plt.tight_layout()
        fig.suptitle('Output of layer{} (Given image{})'.format(cnt, choose_id))
        fig.savefig(os.path.join(base_dir,'layer{}'.format(cnt)))
        
if __name__ == "__main__":
    main()
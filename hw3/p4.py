#!/usr/bin/env python
# -- coding: utf-8 --

import os
import argparse
from keras.models import load_model
from termcolor import colored,cprint
import keras.backend as K
#from utils import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
img_dir = os.path.join(base_dir, 'image')
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
cmap_dir = os.path.join(img_dir, 'cmap')
if not os.path.exists(cmap_dir):
    os.makedirs(cmap_dir)
partial_see_dir = os.path.join(img_dir,'partial_see')
if not os.path.exists(partial_see_dir):
    os.makedirs(partial_see_dir)
model_dir = os.path.join(base_dir, 'model')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

def main():
    parser = argparse.ArgumentParser(prog='plot_saliency.py',
            description='ML-Assignment3 visualize attention heat map.')
    parser.add_argument('--epoch', type=int, metavar='<#epoch>', default=80)
    model_name = "0.67177.h5"
    model_path = os.path.join(model_dir, model_name)
    emotion_classifier = load_model(model_path)
    print(colored("Loaded model from {}".format(model_name), 'yellow', attrs=['bold']))
    
    df = pd.read_csv(base_dir + '/train.csv',encoding="big5")
    rawData = df.as_matrix()
    dfS = len(df)
    private_pixels = [[] for _ in range(dfS)]
    for i in range(0, dfS):
        private_pixels[i] = np.asarray(list(map(int,rawData[i,1].split(' '))))
    private_pixels = np.asarray(private_pixels).astype(float)/255
    del df, rawData
    
    input_img = emotion_classifier.input
    img_ids = [2]

    for idx in img_ids:
        val_proba = emotion_classifier.predict(private_pixels[idx].reshape(1,48,48,1))
        pred = val_proba.argmax(axis=-1)
        target = K.mean(emotion_classifier.output[:, pred])
        grads = K.gradients(target, input_img)[0]
        grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
        fn = K.function([input_img, K.learning_phase()], [grads])

        heatmap = fn([private_pixels[idx].reshape(1,48,48,1),0])
        heatmap = heatmap[0].reshape(48,48)
        '''
        Implement your heatmap processing here!
        hint: Do some normalization or smoothening on grads
        '''
        upBound = 1
        lowBound = 0
        see = private_pixels[idx].reshape(48, 48)
        see[np.where(heatmap <= upBound )and np.where(heatmap >= lowBound)] = np.mean(see)
        
        plt.figure()
        plt.imshow(heatmap, cmap=plt.cm.jet)
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig(os.path.join(cmap_dir, '{}.png'.format(idx)), dpi=100)

        plt.figure()
        plt.imshow(see,cmap='gray')
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig(os.path.join(partial_see_dir, '{}.png'.format(idx)), dpi=100)

if __name__ == "__main__":
    main()
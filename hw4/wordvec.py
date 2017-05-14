#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import word2vec
import nltk
import numpy as np
from sklearn.manifold import TSNE
from adjustText import adjust_text
import matplotlib.pyplot as plt
rootPath = '/Users/apple/desktop/NTUEE/課程八/ML/git/hw4/'

def punc(tag):
    for j in tag:
        if j == ',' or j == '.' or j ==':' or j ==';' or j == '’' or j =='!' or j == '?' or j == '“':
            return False
    return True
        
def selectWord(vec, vocab):
    tagged = nltk.pos_tag(vocab)
    idx,wanted = [],[]
    for i in range(0, len(tagged)):     
        if (len(tagged[i][0]) > 1 and punc(tagged[i][0])) and (tagged[i][1] == 'NN' or tagged[i][1] == 'NNP' or tagged[i][1] == 'NNS' or tagged[i][1] == 'JJ'):
            wanted.append(tagged[i][0])
            idx.append(i)
    vec = vec[idx]
    return vec, wanted

word2vec.word2vec(rootPath + 'hp.txt', rootPath + 'hp.bin', size=100, alpha = 0.025, window = 5, min_count = 5, sample = 0.001)
wordModel = word2vec.load(rootPath + 'hp.bin')
vocab = wordModel.vocab[0:800]
tModel = TSNE(n_components=2, random_state=0)
vec = tModel.fit_transform(wordModel.vectors[0:800]) 
vec, vocab = selectWord(vec, vocab)
    
plt.figure(figsize=(16, 16), dpi=80)
plt.scatter(vec[:, 0], vec[:, 1],s = 2)
texts = []
for x, y, s in zip(vec[:, 0], vec[:, 1],vocab):
    texts.append(plt.text(x,y,s, size = 8))
adjust_text(texts, arrowprops=dict(arrowstyle="-", color='k', lw=0.5))
plt.show()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import csv
import sys
from sklearn.cluster import KMeans
rootPath = '/Users/apple/desktop/NTUEE/課程八/ML/git/hw4/'
arg = True
def elu(arr):
    return np.where(arr > 0, arr, np.exp(arr) - 1)


def make_layer(in_size, out_size):
    w = np.random.normal(scale=0.5, size=(in_size, out_size))
    b = np.random.normal(scale=0.5, size=out_size)
    return (w, b)


def forward(inpd, layers):
    out = inpd
    for layer in layers:
        w, b = layer
        out = elu(out @ w + b)
    return out


def gen_data(dim, layer_dims, N):
    layers = []
    data = np.random.normal(size=(N, dim))

    nd = dim
    for d in layer_dims:
        layers.append(make_layer(nd, d))
        nd = d

    w, b = make_layer(nd, nd)
    gen_data = forward(data, layers)
    gen_data = gen_data @ w + b
    return gen_data


if __name__ == '__main__':
    # if we want to generate data with intrinsic dimension of 10
    #dim = 10
    #N = 10000
    # the hidden dimension is randomly chosen from [60, 79] uniformly
    #layer_dims = [np.random.randint(60, 80), 100]
    #data = gen_data(dim, layer_dims, N)
    # (data, dim) is a (question, answer) pair
    if arg == True:
        rawData = np.load(sys.argv[1])
    else:
        rawData == np.load(rootPath +"data.npz")
    # count Var
    dbVar = np.zeros((200,2))
    for i in range(0,200):
        dbVar[i,0] = np.var(rawData[str(i)])
        dbVar[i,1] = i
    # sort by Var
    sortDb = dbVar[dbVar[:,0].argsort()]
    sortId = sortDb[:,1]
    # K mean cluster
    kmeans = KMeans(n_clusters=60).fit(np.concatenate((np.reshape(sortDb[:,0], (200,1)),np.zeros((200,1))), axis=1))
    clust = kmeans.labels_
    # predict Dimention
    predictD = np.zeros((200,1))
    predictD[0] = 1
    d = 1
    for i in range(1,200):
        if clust[i] != clust[i-1]:
            d += 1 
        predictD[i] = d
    # Sort by Id
    sortDb[:,0] = np.reshape(predictD,(200))      
    sortDb = sortDb[sortDb[:,1].argsort()]
    
    # writeOutput
    parse = []
    parse.append(['SetId','LogDim'])
    for k in range(0,200):  
        parse.append([str(k),str(np.log(sortDb[k,0]))])
    if arg == True:
        pth = sys.argv[2]
    else:
        pth = rootPath +'predictD2.csv'
    with open (pth, 'w') as f:
        writer = csv.writer(f)
        for k in range(0,200+1):
            writer.writerow(parse[k])
    
    
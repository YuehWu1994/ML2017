#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from scipy import misc
from os import listdir
from os.path import isfile, join
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
rootPath = '/Users/apple/desktop/NTUEE/課程八/ML/git/hw4/'
bmpPath = '/Users/apple/desktop/NTUEE/課程八/ML/git/hw4/selectedFace/'
row = 64

def showPicture(pic):
    pcImg = Image.fromarray(pic)
    pcImg.show()

def PCA(pic):
    U, s, Vt = np.linalg.svd(pic,full_matrices=False)
    V = Vt.T
    S = np.diag(s)
    return  U, V, S, s

bmpfiles = [f for f in listdir(bmpPath) if (isfile(join(bmpPath, f)) and f.endswith(".bmp"))]
bmplength = len(bmpfiles)
rawData = np.zeros((100,row**2))
for i in range(0,bmplength):
    rawData[i,:] = np.reshape(misc.imread(os.path.join(bmpPath,bmpfiles[i]), flatten= 0),(1,row**2))


# 1-1 mean of 100 bmp
ave = np.reshape(np.mean(rawData,axis = 0),(1,4096))
avePic = np.reshape(ave, (row,row))
showPicture(avePic)

# 1-1 9 eigenFace
minusAveDb = rawData - ave
minusAveDb = minusAveDb.T
U, V, S, s = PCA(minusAveDb)

for i in range(0,9):
    plt.subplot(3,3,(i+1))
    plt.title(int(S[i,i]), fontsize = 8)
    plt.imshow(np.reshape(U[:,i],(row,row)),cmap = 'gray')
plt.show()
    
# 1-2 ori
new_im = Image.new('F',(800,800))
for i in range(0, bmplength):
    pcImg = Image.fromarray(np.reshape(rawData[i,:], (row,row)))
    new_im.paste(pcImg,(int(i/10)*80, (i%10)*80))
new_im.show()    

# 1-2 5pc
PC = 5
Mhat2 = np.dot(U[:, :PC], np.dot(S[:PC, :PC], V[:,:PC].T))
Mhat2 = (Mhat2.T) + ave
RMSE = np.sqrt(np.mean((rawData - Mhat2)**2))
print ("Using first " + str(PC) + " PCs, RMSE = %.6G" %RMSE)
for i in range(0, bmplength):
    pcImg = Image.fromarray(np.reshape(Mhat2[i,:], (row,row)))
    new_im.paste(pcImg,(int(i/10)*80, (i%10)*80))
new_im.show()   
    


    

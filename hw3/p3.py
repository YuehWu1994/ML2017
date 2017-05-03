#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
from keras.models import Sequential, load_model

modelNum = 10
rootPath = '/Users/apple/desktop/NTUEE/課程八/ML/git/hw3/'

model2 = load_model("0.67177.h5")
df = pd.read_csv(rootPath + 'train.csv',encoding="big5")
rawData = df.as_matrix()
dfS = len(df)
feaX = [[] for _ in range(dfS)]
feaY = np.zeros((dfS,7))
 
for i in range(0, dfS):
    feaX[i] = np.asarray(list(map(int,rawData[i,1].split(' '))))
    feaY[i,int(rawData[i,0])] = 1
feaX = np.asarray(feaX).astype(float)/255
del df, rawData
x_test = feaX[int(np.ceil(dfS/modelNum*0)):int(np.ceil(dfS/modelNum*(1)))]
x_test = x_test.reshape(x_test.shape[0],48,48,1)
y_test = feaY[int(np.ceil(dfS/modelNum*0)):int(np.ceil(dfS/modelNum*(1)))]
y_true = np.argmax(y_test, axis = 1)

pdWeight = model2.predict(x_test)
y_pred = np.argmax(pdWeight, axis = 1)

c = confusion_matrix(y_true, y_pred)
class_names = ['angry','hatred','fear','happy','sad','surprise','neutral']

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm,decimals=2)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    
    print(cm)

   

    thresh = cm.max() / 2.
           
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(c, classes=class_names,title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(c, classes=class_names, normalize=True, title='Normalized confusion matrix')

plt.show()


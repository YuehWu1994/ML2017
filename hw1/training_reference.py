import numpy as np
import pandas as pd
import random
import sys
import os
import csv
from numpy import genfromtxt
#rootPath = '/home/b02901093/'
rootPath = '/Users/apple/desktop/NTUEE/課程八/ML/git/hw1/'
# global var
startPoint = 3
numFeature = 18
trainFea = 162
ans = numFeature + trainFea - 9
notCount = 9
featureRow = 0
testRow = 0
wantedWeight = np.asarray([[-0.01705091, -0.02083142,  0.21253986, -0.23651794, -0.04069132,  0.52401146 ,
 -0.57632458,  0.01898099,  1.10208739,  0.28503703],[ 0.00259471, -0.04402418,  0.20843701, -0.21511481, -0.04312318,  0.50434947,
 -0.55964028,  0.02054308,  1.09859731,  0.08996173],[ -1.09633689e-02,  -4.19719474e-02,   2.45442519e-01,  -2.46912439e-01,
  -6.06590459e-02,   5.50714950e-01,  -5.79461522e-01 ,  2.37820184e-05,
   1.11779276e+00,   4.95601020e-02],[-0.00352962, -0.04501271,  0.2270545,  -0.21106619, -0.09540268,  0.55710901,
 -0.55883817, -0.02183229,  1.12610007,  0.03398316]])
featureSet = []
testSet = []
gradient = []
lost = 0
randSet = []
setGradient = []
learningRate = 0.0001
models = 4
overFit = 0  # 0 -> Testing, 1 -> noTesting, 2 -> use existing weight

# function
def feaSelect(featureRow, trainFea, ans):
    numFea = 1
    tmpSet = np.zeros((featureRow, int(trainFea/numFeature) * numFea + 1))
    pm2_5 = 9
    for i in range(0, featureRow):
         tmpSet[i,0:9:1] = featureSet[i][pm2_5:162:18]
         tmpSet[i,9] = featureSet[i][ans]
    return tmpSet
    
def genRandom(randSet):
    for i in range(0, trainFea + 1):
        randSet.append(float(random.uniform(-1,1)))

def sepFeature(row,column,data):
    days = int(row/numFeature)
    for d in range(0, days):
        for c in range(startPoint, column):
            for i in range(0, numFeature):
                tmp = data[d * numFeature + i][c]
                if tmp == 'NR':
                    tmp = 0
                featureSet.append(float(tmp))

def sepFeatureTest(row,column,data):
    days = int(row/numFeature)
    for d in range(0, days):
        for c in range(2, column):
            for i in range(0, numFeature):
                tmp = data[d * numFeature + i][c]
                if i == 10:
                    tmp = 0
                featureSet.append(float(tmp))
                
def loopModel(sub1,sub2,sub3):
    featureSet = np.append(sub1,sub2, axis=0)
    if overFit != 0:
        featureSet = np.append(featureSet,sub3, axis=0)
    testSet = sub3
    featureRow = int(featureSet[:,0].size)  
    return featureSet, testSet, featureRow
            
def stdScalar():
    for r in range(0, featureRow):
        std = np.std(featureSet[r,0:trainFea])
        mean = np.mean(featureSet[r,0:trainFea])
        for i in range(0, trainFea):
            if std != 0:
                featureSet[r,i] = (featureSet[r,i] - mean)/std          
def countLost(): 
    lost = 0
    for r in range(0,featureRow):
        estAns = sigma(r)
        lost += (featureSet[r,ans] - estAns)**2
    print(lost)
    return lost

def countLostTest(): 
    lost = 0
    for r in range(0,testRow):
        estAns = testSigma(r)
        lost += (testSet[r,ans] - estAns)**2
    #print(lost)
    return lost

def setGradient():
    for i in range(0, trainFea + 1):        
        if i == trainFea:
            grad = diffBias(i)
        else:
            grad = diff(i)    
        gradient.append(grad)

def diff(idx):
    diffVal = 0
    for r in range(0,featureRow):
        estAns = sigma(r)
        diffVal += (featureSet[r,ans] - estAns)* 2 * (-featureSet[r,idx])
    return diffVal

def diffBias(idx):
    diffVal = 0
    for r in range(0,featureRow):
        estAns = sigma(r)
        diffVal += (featureSet[r,ans] - estAns)* 2
    return diffVal
    
def sigma(r):
    estAns = 0
    for i in range(0, trainFea):
        estAns += featureSet[r,i] * randSet[i]
    estAns += randSet[trainFea]
    return estAns

def testSigma(r):
    estAns = 0
    for i in range(0, trainFea):
        estAns += testSet[r,i] * randSet[i]
    estAns += randSet[trainFea]
    return estAns

def adagrad():
    row = int(arrGradient.size/arrGradient[0].size)
    gradRootSqr = []
    for i in range(0, row):
        gradRootSqr.append(np.sqrt(np.sum(arrGradient[i]**2)))
    gradRootSqr = np.asarray(gradRootSqr)
    return gradRootSqr
       
def changeWeight(gradRootSqr):
    return (randSet - learningRate/gradRootSqr * gradient)

def duplicatePM2_5(r):
    r = np.reshape(np.asarray(r),(-1,1))
    tmpSet = []
    tmpSize = r.size
    tmpWanted = 24*19+15
    for i in range(0, tmpSize):
        if i % 480 < tmpWanted:
            tmpSet.append(r[i:i+10])
    lengthTmp = len(tmpSet)                     
    for i in range(0, lengthTmp):
        tmpSet[i] = np.array(tmpSet[i]).ravel()
    return np.asarray(tmpSet[:2952])                      # mod here
    
    
# main - init
os.chdir(rootPath)
df = pd.read_csv("train.csv",encoding="big5")
rawData = df.as_matrix()
column = int(rawData[0].size)
row = len(df)

sepFeature(row,column,rawData)
del rawData
del df 
featureSet = featureSet[0:int(len(featureSet)/(trainFea+numFeature))*(trainFea+numFeature)]
featureSet = np.reshape(np.asarray(featureSet),(-1,(trainFea+numFeature))) 
featureRow = int(featureSet[:,0].size)
# select PM2.5 only

featureSet = feaSelect(featureRow, trainFea, ans)
ans = featureSet[0].size -1
trainFea = ans
featureSet = duplicatePM2_5(featureSet)   

sub1 = featureSet[0:984]
sub2 = featureSet[984:1968]
sub3 = featureSet[1968:2952]
testRow = int(sub1[:,0].size)  
lostSet = np.zeros((3, models))
#stdScalar()

if overFit == 0:
    for idx in range(0,3):
        if idx == 0:
            featureSet, testSet, featureRow = loopModel(sub1,sub2,sub3)
        elif idx == 1:
            featureSet, testSet, featureRow = loopModel(sub3,sub1,sub2)
        else :
            featureSet, testSet, featureRow = loopModel(sub3,sub2,sub1)
        for i in range(0,models): 
            # randomSet
            countLoop = 0
            randSet = []
            randSet = wantedWeight[i]  
            # countLost & diffrentiate
            lost = countLost()
            setGradient()
            gradient = np.asarray(gradient)
            arrGradient = gradient       
            gradRootSqr = adagrad()
            randSet = changeWeight(gradRootSqr)
            count = 0
            while (count < 2):   
                gradient = []
                nxtlost = countLost()
                #print(countLoop)
                if countLoop == 10 and nxtlost > 600000:
                    break
                if countLoop == 40 and nxtlost > 300000:                
                    break
                countLoop += 1
                if nxtlost > lost:
                    count += 1
                else:
                    count == 0
                if count == 2:
                    #print(lost)
                    print('===Testing===')
                    testLost = countLostTest()
                    lostSet[idx,i] = testLost
                    print(str(i+1) + ' ' + str(testLost))
                    print(randSet)
                    break       
                lost = nxtlost
                setGradient()
                gradient = np.asarray(gradient)
                arrGradient = np.column_stack((arrGradient,gradient))
                gradRootSqr = adagrad()
                randSet = changeWeight(gradRootSqr)
        if idx == 0:
            wantedWeight = np.asarray(wantedWeight)
    minMean = sys.maxsize 
    modelIdx = 0       
    for i in range(0,models):
        if np.mean(lostSet[:,i]) < minMean:
            minMean = np.mean(lostSet[:,i])
            modelIdx = i
    print('MODEL : ' + str(modelIdx))
    print (wantedWeight[modelIdx])

elif overFit == 1:
    featureSet, testSet, featureRow = loopModel(sub1,sub2,sub3)
    for i in range(0,models): 
        # randomSet
        countLoop = 0
        randSet = []
        genRandom(randSet)
        randSet = np.asarray(randSet)
        # countLost & diffrentiate
        lost = countLost()
        setGradient()
        gradient = np.asarray(gradient)
        arrGradient = gradient       
        gradRootSqr = adagrad()
        randSet = changeWeight(gradRootSqr)
        count = 0
        while (count < 2):   
            gradient = []
            nxtlost = countLost()
            #print(countLoop)
            if countLoop == 10 and nxtlost > 780000:
                break
            if countLoop == 40 and nxtlost > 390000:                
                break
            if countLoop == 200 and nxtlost > 240000:  
                break
            if countLoop >= 200 and nxtlost> 240000:
                break
            if countLoop == 400 and nxtlost > 222000:
                break
            if countLoop >= 400 and nxtlost > 222000:
                break
            countLoop += 1
            if nxtlost > lost:
                count += 1
            else:
                count == 0
            if count == 2:
                #print(lost)
                #testLost = countLostTest()
                lostSet[0,i] = lost
                if lost < 220000:
                    print(str(i+1) + ' ' + str(lost))
                    print(randSet)
                wantedWeight.append(randSet)
                break       
            lost = nxtlost
            setGradient()
            gradient = np.asarray(gradient)
            arrGradient = np.column_stack((arrGradient,gradient))
            gradRootSqr = adagrad()
            randSet = changeWeight(gradRootSqr)
    wantedWeight = np.asarray(wantedWeight)
    minMean = sys.maxsize 
    modelIdx = 0       
    for i in range(0,models):
        if np.mean(lostSet[:,i]) < minMean:
            minMean = np.mean(lostSet[:,i])
            modelIdx = i
    print('MODEL : ' + str(modelIdx))
    print (wantedWeight[modelIdx])
    
else:
    featureSet, testSet, featureRow = loopModel(sub1,sub2,sub3)
    randSet = [[-0.0249752,   0.0124962,   0.19656032, -0.23835816, -0.03681029,0.52964621,-0.57331572,  0.00564422,  1.11983566, -0.46484609],[-0.00770649, -0.01909839,  0.19983943, -0.2196687,  -0.0355595,   0.4991455, -0.55196926,  0.00756328 , 1.11218517, -0.28400761],[-0.00490165, -0.02656738,  0.20534024, -0.21437064 ,-0.05218807,  0.50885927, -0.5403249,  -0.00753782,  1.11684495, -0.29433414],[-0.00173309, -0.0421378,   0.23791051, -0.24121027, -0.06070744,  0.56402171,
 -0.59896725,  0.0074344 ,  1.12597481, -0.4577249 ],[ 0.0037982,  -0.05316117,  0.23152179, -0.22588066, -0.06510202,  0.5441635,
 -0.57568226 , 0.00645916,  1.11575171, -0.19069656],[-0.00352962, -0.04501271,  0.2270545,  -0.21106619, -0.09540268,  0.55710901,
 -0.55883817, -0.02183229,  1.12610007,  0.03398316]]
    randSet = np.asarray(randSet)
    # countLost & diffrentiate
    lost = countLost()
    setGradient()
    gradient = np.asarray(gradient)
    arrGradient = gradient       
    gradRootSqr = adagrad()
    randSet = changeWeight(gradRootSqr)
    count = 0
    while (count < 2):   
        gradient = []
        nxtlost = countLost()
        
        if nxtlost > lost:
            count += 1
        else:
            count == 0
        if count == 2:
            print(randSet)
            wantedWeight.append(randSet)
            break       
        lost = nxtlost
        setGradient()
        gradient = np.asarray(gradient)
        arrGradient = np.column_stack((arrGradient,gradient))
        gradRootSqr = adagrad()
        randSet = changeWeight(gradRootSqr)    
    wantedWeight = np.asarray(wantedWeight)
    minMean = sys.maxsize 
    modelIdx = 0       


# TESTING
featureSet = []
#df_test = pd.read_csv("test_X.csv",encoding="big5")
testData = genfromtxt("test_X.csv",dtype=None,delimiter=',')
column = int(testData[0].size)
row = len(testData)
sepFeatureTest(row,column,testData)
del testData
trainFea = 162   
featureSet = np.reshape(np.asarray(featureSet),(-1,trainFea)) 
featureRow = int(featureSet[:,0].size)

featureSet = feaSelect(featureRow, trainFea, ans)
ans = featureSet[0].size -1
trainFea = ans

predict = []
for r in range (0,featureRow):
    estAns = 0
    for i in range(0, 9):
        estAns += featureSet[r,i] * wantedWeight[modelIdx,i]
    estAns += wantedWeight[modelIdx,9]
    predict.append(estAns)
parse = []
parse.append(['id','value'])
for k in range(0,240):
    parse.append(['id_' + str(k), predict[k]])

with open (rootPath + "submi.csv", 'w') as f:
    writer = csv.writer(f)
    for k in range(0,241):
        writer.writerow(parse[k])


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.ensemble import RandomForestClassifier

import idx2numpy

def readMnist(filename):
    arr = idx2numpy.convert_from_file(filename)
    return arr

def flattenMnist(mnistImage):
    nRecords = len(mnistImage)
    return np.reshape(mnistImage, [nRecords,28*28]).astype(int)

def addNoiseToData(data, nNoisyDimToAdd):
    nRecords = len(data)
    noisyData = np.random.uniform(low=0, high=255, size=(nRecords,nNoisyDimToAdd))
    noiseAddedData = np.concatenate((data, noisyData), axis=1)
    return noiseAddedData

def getTreeAccuracy(treeObj, XTest, YTest):
    nRecords = len(YTest)
    predictedY = treeObj.predict(XTest)
    accuracy = np.sum(predictedY==YTest) / nRecords
    return accuracy

"""
READ DATA
"""
trainImages = readMnist("train-images.idx3-ubyte")
trainLabels = readMnist("train-labels.idx1-ubyte").astype(int)

"""
PREPROCESS
"""
trainData = flattenMnist(trainImages)

"""
DEFINE NUMBER OF IRRELEVANT DIMENSIONS TO ADD IN EACH RUN
"""
nNoiseToAdd = 50

"""
PREPROCESS
"""
noiseAddedTrainSet = addNoiseToData(trainData, nNoiseToAdd)

nFeatures = 28*28+nNoiseToAdd
n_estimators = int(nFeatures/3)

# Fit Random Forest
forest = RandomForestClassifier(n_estimators=n_estimators, oob_score=True, n_jobs=-1)
forest.fit(noiseAddedTrainSet, trainLabels)

"""
POSTPROCESS
"""
importances = forest.feature_importances_
importances = np.reshape(importances,(nFeatures,1))
indices = np.array(list(range(nFeatures))).reshape((nFeatures,1))
indicesVSImportance = np.concatenate((indices,importances), axis=1)
sortedImportances = np.array(sorted(indicesVSImportance, key=lambda x: x[1], reverse=True))

dataToCalcSTD = [tree.feature_importances_ for tree in forest.estimators_]
std = np.std(dataToCalcSTD, axis=0).reshape((nFeatures,1))
indices = np.array(list(range(nFeatures))).reshape((nFeatures,1))
indicesVSSTDImportance = np.concatenate((indices,std), axis=1)
sortedSTDImportances = np.array(sorted(indicesVSSTDImportance, key=lambda x: x[1], reverse=True))

noisyFeatureX = list(range(28*28, nFeatures))
noisyFeatureY = sortedSTDImportances[noisyFeatureX,1]


"""
PLOTS
"""

plt.figure(figsize=(8,8))
plt1 = plt.plot(list(range(nFeatures)), sortedImportances[:,1], linewidth=3.0)
plt2 = plt.plot(list(range(nFeatures)), sortedSTDImportances[:,1], linewidth=3.0)
plt3 = plt.scatter(noisyFeatureX, noisyFeatureY, s=100, c="red")

plt.title("Mean Importance And\n STD Of Importance Of Each Feature", fontsize=16)
plt.xlabel("Features (With 50 Noisy Features)", fontsize=16)
plt.ylabel("Mean Importance and STD Of Importance", fontsize=16)
blue_patch = mpatches.Patch(color='blue', label='Mean Feature Importance')
orange_patch = mpatches.Patch(color='orange', label='STD Of Features Importance')
red_patch = mpatches.Patch(color='red', label='Noisy Features')
plt.legend(handles=[blue_patch, orange_patch, red_patch])
plt.show()

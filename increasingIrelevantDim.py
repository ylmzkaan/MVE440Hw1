

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

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
testImages = readMnist("t10k-images.idx3-ubyte")
testLabels = readMnist("t10k-labels.idx1-ubyte").astype(int)

NUMBER_OF_DATA_POINTS_IN_TRAINDATA = 10000
trainImages = trainImages[:NUMBER_OF_DATA_POINTS_IN_TRAINDATA]
trainLabels = trainLabels[:NUMBER_OF_DATA_POINTS_IN_TRAINDATA]

"""
PREPROCESS
"""
trainData = flattenMnist(trainImages)
testData = flattenMnist(testImages)


"""
DEFINE NUMBER OF IRRELEVANT DIMENSIONS TO ADD IN EACH RUN
"""
nNoiseToAddList = list(range(0,2000,100))
size_NNoiseToAddList = len(nNoiseToAddList)

"""
DATA TO SAVE
"""
oobError = np.empty((size_NNoiseToAddList))
rfAccuracy = np.empty((size_NNoiseToAddList))
dtAccuracy = np.empty((size_NNoiseToAddList))

"""
RUN SIMULATIONS
"""
for i, nNoisyDimToAdd in enumerate(nNoiseToAddList):
    print("Number of noisy dimensions to add: {}".format(nNoisyDimToAdd))
    noiseAddedTrainSet = addNoiseToData(trainData, nNoisyDimToAdd)
    noiseAddedTestSet = addNoiseToData(testData, nNoisyDimToAdd)

    n_estimators = int((28*28+nNoisyDimToAdd)/3)

    # Fit Random Forest
    forest = RandomForestClassifier(n_estimators=n_estimators, oob_score=True, n_jobs=-1)
    forest.fit(noiseAddedTrainSet, trainLabels)

    oobError[i] = 1 - forest.oob_score_
    rfAccuracy[i] = getTreeAccuracy(forest, noiseAddedTestSet, testLabels)

    # Fit Decision Tree
    clf = DecisionTreeClassifier()
    clf = clf.fit(noiseAddedTrainSet, trainLabels)

    dtAccuracy[i] = getTreeAccuracy(clf, noiseAddedTestSet, testLabels)


"""
PLOTS
"""
plt.figure()
pltOOBError = plt.plot(nNoiseToAddList, oobError, linewidth=3.0)
plt.title("OOB Error for Random Forest\n With Increasing Irrelevant Dimensions", fontsize=16)
plt.xlabel("Number of Irrelevant Dimensions", fontsize=16)
plt.ylabel("Error", fontsize=16)
plt.ylim(bottom=0, top=0.2)
blue_patch = mpatches.Patch(color='blue', label='Random Forest OOB Error')
plt.legend(handles=[blue_patch])
plt.show()

plt.figure()
plt1 = plt.plot(nNoiseToAddList, rfAccuracy, linewidth=3.0)
plt2 = plt.plot(nNoiseToAddList, dtAccuracy, linewidth=3.0)
plt.title("Accuracy for Random Forest And Decision Tree\n With Increasing Irrelevant Dimensions", fontsize=16)
plt.xlabel("Number of Irrelevant Dimensions", fontsize=16)
plt.ylabel("Accuracy", fontsize=16)
plt.ylim(bottom=0.7, top=1)
blue_patch = mpatches.Patch(color='blue', label='Random Forest Accuracy')
orange_patch = mpatches.Patch(color='orange', label='Decision Tree Accuracy')
plt.legend(handles=[blue_patch, orange_patch])
plt.show()

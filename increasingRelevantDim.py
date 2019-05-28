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

def isolateNMostImportantFeatures(data, featureImportances, nDimsToIsolate):
    if nDimsToIsolate != 0:
        nFeatures = len(featureImportances)

        featureImportances = np.reshape(featureImportances, (nFeatures,1))
        featureIndices = np.reshape(np.array(list(range(nFeatures))), (nFeatures,1))

        featureImportances = np.concatenate((featureIndices, featureImportances), axis=1)
        sortedFeatureImportances = np.array(sorted(featureImportances, key=lambda x: x[1], reverse=True))

        indicesOfFeaturesToDelete = sortedFeatureImportances[nDimsToIsolate:,0]

        data = np.delete(data, indicesOfFeaturesToDelete, axis=1)
    return data

def getTreeAccuracy(treeObj, XTest, YTest):
    nRecords = len(YTest)
    predictedY = treeObj.predict(XTest)
    accuracy = np.sum(predictedY==YTest) / nRecords
    return accuracy

featureImportances = np.load("featureImportances.npy")
nFeaturesDefault = 28*28

"""
GET DATA
"""
trainImages = readMnist("train-images.idx3-ubyte")
trainLabels = readMnist("train-labels.idx1-ubyte").astype(int)
testImages = readMnist("t10k-images.idx3-ubyte")
testLabels = readMnist("t10k-labels.idx1-ubyte").astype(int)

"""
PREPROCESSING
"""
trainData = flattenMnist(trainImages)
testData = flattenMnist(testImages)

"""
SIMULATION SPECIFIC SETTINGS
"""
nMostImportantFeaturesToIsolateList = list(range(1, 784, 20))
size_nMostImportantFeaturesToIsolateList = len(nMostImportantFeaturesToIsolateList)

"""
RESULTS TO PLOT LATER
"""
oobError = np.empty((size_nMostImportantFeaturesToIsolateList))
rfAccuracy = np.empty((size_nMostImportantFeaturesToIsolateList))
dtAccuracy = np.empty((size_nMostImportantFeaturesToIsolateList))

for i, nDimsToIsolate in enumerate(nMostImportantFeaturesToIsolateList):

    # Delete relevant dims
    trainSetWithDimsDeleted = isolateNMostImportantFeatures(trainData, featureImportances, nDimsToIsolate)
    testSetWithDimsDeleted = isolateNMostImportantFeatures(testData, featureImportances, nDimsToIsolate)

    # Number of trees
    n_estimators = max(int(nDimsToIsolate/3),30)

    # Fit random forest
    forest = RandomForestClassifier(n_estimators=n_estimators, oob_score=True, n_jobs=-1)
    forest.fit(trainSetWithDimsDeleted, trainLabels)

    # Save random forest results
    oobError[i] = 1 - forest.oob_score_
    rfAccuracy[i] = getTreeAccuracy(forest, testSetWithDimsDeleted, testLabels)

    # Fit decision tree
    clf = DecisionTreeClassifier()
    clf = clf.fit(trainSetWithDimsDeleted, trainLabels)

    # Save DT result
    dtAccuracy[i] = getTreeAccuracy(clf, testSetWithDimsDeleted, testLabels)


"""
PLOT
"""

plt.figure()
pltOOBError = plt.plot(nMostImportantFeaturesToIsolateList, oobError, linewidth=3.0)
plt.title("OOB Error for Random Forest\n With Increasing Relevant Dimensions", fontsize=16)
plt.xlabel("Number of Most Important Features Fed To Model", fontsize=16)
plt.ylabel("Error", fontsize=16)
plt.ylim(bottom=0, top=0.9)
blue_patch = mpatches.Patch(color='blue', label='Random Forest OOB Error')
plt.legend(handles=[blue_patch])
plt.show()

plt.figure()
plt1 = plt.plot(nMostImportantFeaturesToIsolateList, rfAccuracy, linewidth=3.0)
plt2 = plt.plot(nMostImportantFeaturesToIsolateList, dtAccuracy, linewidth=3.0)
plt.title("Accuracy for Random Forest And Decision Tree\n With Increasing Relevant Dimensions", fontsize=16)
plt.xlabel("Number of Most Important Features Fed To Model", fontsize=16)
plt.ylabel("Accuracy", fontsize=16)
plt.ylim(bottom=0.1, top=1)
blue_patch = mpatches.Patch(color='blue', label='Random Forest Accuracy')
orange_patch = mpatches.Patch(color='orange', label='Decision Tree Accuracy')
plt.legend(handles=[blue_patch, orange_patch])
plt.show()

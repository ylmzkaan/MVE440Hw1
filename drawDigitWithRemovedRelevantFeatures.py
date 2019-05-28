import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

import idx2numpy

def readMnist(filename):
    arr = idx2numpy.convert_from_file(filename)
    return arr

def flattenMnist(mnistImage):
    nRecords = len(mnistImage)
    return np.reshape(mnistImage, [nRecords,28*28]).astype(int)

def takeNMostImportantFeatures(data, featureImportances, nDimsToIsolate):
    if nDimsToIsolate != 0:
        nFeatures = len(featureImportances)

        featureImportances = np.reshape(featureImportances, (nFeatures,1))
        featureIndices = np.reshape(np.array(list(range(nFeatures))), (nFeatures,1))

        featureImportances = np.concatenate((featureIndices, featureImportances), axis=1)
        sortedFeatureImportances = np.array(sorted(featureImportances, key=lambda x: x[1], reverse=True))

        indicesOfFeaturesToDelete = sortedFeatureImportances[nDimsToIsolate:,0].astype(int)

        data[:,indicesOfFeaturesToDelete] = 0
    return data

"""
READ DATA
"""
trainImages = readMnist("train-images.idx3-ubyte")
trainLabels = readMnist("train-labels.idx1-ubyte").astype(int)

"""
PREPROCESS DATA
"""

featureImportances = np.load("featureImportances.npy")

trainData = flattenMnist(trainImages)

nDimsToIsolate = 100
trainData = takeNMostImportantFeatures(trainData, featureImportances, nDimsToIsolate)

idxDigitToDraw = 12000
img = np.reshape(trainData[idxDigitToDraw, :], [28,28])
plt.figure()
plt.imshow(img)
plt.title("Digit {} With {} Most Important Features".format(trainLabels[idxDigitToDraw], nDimsToIsolate))
plt.show()

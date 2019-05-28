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

def addNoiseToData(data, nNoisyDimToAdd):
    if nNoisyDimToAdd == 0:
        return data
    nRecords = len(data)
    noisyData = np.random.uniform(low=0, high=255, size=(nRecords,nNoisyDimToAdd))
    noiseAddedData = np.concatenate((data, noisyData), axis=1)
    return noiseAddedData

"""
READ DATA
"""
trainImages = readMnist("train-images.idx3-ubyte")
trainLabels = readMnist("train-labels.idx1-ubyte").astype(int)

"""
PREPROCESS DATA
"""
# It is better if this is a multiplier of 28 (Digit drawing purposes)
nNoisyDimToAdd = 0 # If zeros then it is a regular RF

trainData = flattenMnist(trainImages)
trainData = addNoiseToData(trainData, nNoisyDimToAdd)

"""
FIT RANDOM FOREST
"""
n_estimators = int(28*28/3)
forest = RandomForestClassifier(n_estimators=n_estimators,n_jobs=-1)
forest.fit(trainData, trainLabels)

"""
GET IMPORTANCE AND MANIPULATE
"""
importances = forest.feature_importances_

"""
if nNoisyDimToAdd == 0:
    np.save("featureImportances2.npy", importances)
"""

dataToCalcSTD = [tree.feature_importances_ for tree in forest.estimators_]
std = np.std(dataToCalcSTD, axis=0)

nFeatures = 28*28 + nNoisyDimToAdd
importances = np.reshape(importances,(nFeatures,1))
indices = np.array(list(range(nFeatures))).reshape((nFeatures,1))
indicesVSImportance = np.concatenate((indices,importances), axis=1)
sortedImportances = np.array(sorted(indicesVSImportance, key=lambda x: x[1], reverse=True))
indicesOfMostImportant20Feature = sortedImportances[:20,0].astype(int)

stdHead20 = std[indicesOfMostImportant20Feature]
importanceHead20 = importances[indicesOfMostImportant20Feature]

# Plot Heatmap
try:
    nRows = 28 + int(nNoisyDimToAdd/28)
    importanceImage = np.resize(importances, (nRows,28))

    plt.figure()
    plt.imshow(importanceImage)
    plt.colorbar();
    plt.title("Feature Importance", fontsize=16)
except:
    print("Could not generate a heat map because number of features is" +
          "not appropriate to reshape and form a meaningful digit")

# Plot importance bar chart
plt.figure()
plt.bar(list(range(nFeatures)), importances[:,0])
plt.title("Feature Importances", fontsize=16)
plt.xlabel("Features", fontsize=16)
plt.ylabel("Importance", fontsize=16)
plt.show()


# Draw importance with error bar
stdHead20 = np.reshape(stdHead20,(20,1))

plt.figure()
plt.title("Feature Importances \n(Error bars are standard devations)", fontsize=16)
plt.xlabel("Most Important 20 Features", fontsize=16)
plt.ylabel("Gini Importance", fontsize=16)
plt.bar(list(range(20)), importanceHead20[:,0], color="r", yerr=stdHead20, align="center")
plt.xlim([-1, 20])
plt.ylim([0, 0.1])
plt.show()


# Plot cumulative feature importance
x_values = list(range(nFeatures))
plt.figure(figsize=(8,8))
plt.plot(x_values, np.cumsum(importances), 'g-', linewidth=3.0)
# Draw line at 95% of importance retained
plt.hlines(y = 0.95, xmin=0, xmax=len(importances), color = 'r',
           linestyles = 'dashed', linewidth=3.0)
# Format x ticks and labels
plt.xticks(np.arange(0,nFeatures,step=20), rotation = 'vertical')
plt.annotate('95%', xy=(10, 0.95), xytext=(5, 0.96))
# Axis labels and title
plt.xlabel('Features', fontsize=16);
plt.ylabel('Cumulative Importance', fontsize=16);
plt.title('Cumulative Importances', fontsize=16);
plt.show()

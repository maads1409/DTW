#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 02:00:47 2018

@author: Mohammed
"""

# K-Nearest Neighbors (K-NN) classifier with DTW using 10-fold cross-validation

# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
import math
import timeit

totalCostComputationTime = []
totalPathComputationTime = []
step = 0

# Euclidean distance.
def mdist(x,y):
    #return abs(x-y)
    return math.sqrt((x-y)**2)


# Calculate DTW distance.
#def dtwDistance(s, t, sakoe_chiba_band_percentage=100):
def dtwDistance(s, t, sakoe_chiba_band_percentage=100):
    n,m = len(s), len(t)
    #sakoe_chiba_band = max(n, m) * sakoe_chiba_band_percentage / 100.0
    n = n+1
    m = m+1
    dtw = [[math.inf for x in range(m)] for y in range(n)]
    dtw[0][0] = 0

    global totalCostComputationTime, totalPathComputationTime, step
    beginTime = timeit.default_timer()

    midTime = timeit.default_timer()
    for i in range(1,n):
        for j in range(1, m):
            dtw[i][j] = mdist(s[i-1],t[j-1]) + min(dtw[i-1][j], dtw[i][j-1], dtw[i-1][j-1])

    totalPathComputationTime[step] = totalPathComputationTime[step] + timeit.default_timer() - midTime
    totalCostComputationTime[step] = totalCostComputationTime[step] + midTime - beginTime
    return dtw[n-1][m-1]


# Knn classifier function with k-fold validation
#def KnnClassifier(X, y, K=10, sakoe_chiba_band_percentage=100):
def KnnClassifier(X, y, K=10):
    #K-Folds cross-validator
    kf = StratifiedKFold(n_splits=K)

    # Outputs
    confusionMatrices = []
    classifierReports = []
    accuracies = np.zeros(K)
    runtimes = np.zeros(K)
    global totalCostComputationTime, totalPathComputationTime, step
    totalCostComputationTime = np.zeros(K)
    totalPathComputationTime = np.zeros(K)

    for step, (train_index, test_index) in enumerate(kf.split(X, y)):
        # RunTime
        print("{:.2f}% done".format(step*100/K))
        beginTime = timeit.default_timer()

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        #classifier = KNeighborsClassifier(n_neighbors = 1, metric = lambda s, t: dtwDistance(s, t, sakoe_chiba_band_percentage))
        classifier = KNeighborsClassifier(n_neighbors = 1, metric = lambda s, t: dtwDistance(s, t))
        classifier.fit(X_train, y_train)

        # Predicting the Test set results
        y_pred = classifier.predict(X_test)

        # Making the Confusion Matrix
        confusionMatrices.append(confusion_matrix(y_test, y_pred))
        classifierReports.append(classification_report(y_test, y_pred))

        accuracies[step] = accuracy_score(y_test, y_pred)
        runtimes[step] = timeit.default_timer() - beginTime

    return confusionMatrices, classifierReports, accuracies, runtimes


def main():
    # Import dataset
    dataset_Test = pd.read_csv('C_OliveOil.txt')
    X = dataset_Test.iloc[:,1:].values
    y = dataset_Test.iloc[:,0].values

    #confusionMatrices, classifierReports, accuracies, runtimes  = KnnClassifier(X, y, sakoe_chiba_band_percentage=1)
    confusionMatrices, classifierReports, accuracies, runtimes  = KnnClassifier(X, y)
    tCCT = 0
    tPCT = 0
    rT = 0
    print ("100% done")

    for i in range(len(confusionMatrices)):
        print("Results of fold #{:d}".format(i))
        print(confusionMatrices[i])
        print(classifierReports[i])
        print("Accuracy = {:.2f}".format(accuracies[i]))
        print("Run time: {0:.4f}".format(runtimes[i]))
        print("Cost Computation time: {0:.4f}".format(totalCostComputationTime[i]))
        print("Path Computation time: {0:.4f}\n".format(totalPathComputationTime[i]))
        tCCT = tCCT + totalCostComputationTime[i]
        tPCT = tPCT + totalPathComputationTime[i]
        rT = rT + runtimes[i]

    accuracyMean = accuracies.mean() * 100
    accuracyStd = accuracies.std()
    print ('Mean Classifier Accuracy = {0:.2f}'.format(accuracyMean))
    print ('Std Classifier Accuracy = {0:.3f}.'.format(accuracyStd))
    print("Total Run time: {0:.4f}".format(rT))
    print("Total Cost Computation time: {0:.4f}".format(tCCT))
    print("Total Path Computation time: {0:.4f}\n".format(tPCT))

if __name__ == "__main__":
    main()
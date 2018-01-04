import cv2
import numpy as np
from os import path
import cPickle as pickle
from matplotlib import pyplot as plt
"""Load preprocessed faces from pickle file or create them from scratch.
Pickle file contains train and test images, corresponding labels of train / test images,
eigenvectors and mean values"""

def loadData(loadFromFile, splittingFactor = 0.2,
             saveToFile = None):
    seed = 1234
    X = []
    labels = []
    if not path.isfile(loadFromFile):
        print"File not found ", loadFromFile
        return (X, labels), (X, labels), None, None
    else:
        file = open(loadFromFile, 'rb')
        samples = pickle.load(file)
        labels = pickle.load(file)
        print "Loaded", len(samples), "training samples"

        X, V, m = extractFeatures(samples)

        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(labels)

        xTrain = X[:int(len(X)*(1-splittingFactor))]
        yTrain = labels[:int(len(X)*(1-splittingFactor))]

        xTest = X[int(len(X)*(1-splittingFactor)):]
        yTest = labels[int(len(X)*(1-splittingFactor)):]

        if saveToFile is not None:
            file = open(saveToFile, 'wb')
            pickle.dump(xTrain, file)
            pickle.dump(yTrain, file)
            pickle.dump(xTest, file)
            pickle.dump(yTest, file)
            pickle.dump(V, file)
            pickle.dump(m, file)
            file.close()
        return (xTrain, yTrain), (xTest, yTest), V, m


def loadPreprocessedData(file):
    if path.isfile(file):
        f = open(file)
        xTrain = pickle.load(f);
        yTrain = pickle.load(f)
        xTest = pickle.load(f)
        yTest = pickle.load(f)
        V = pickle.load(f)
        m = pickle.load(f)
        f.close()

        return (xTrain, yTrain), (xTest, yTest), V, m


def extractFeatures(X, V=None, m=None):
    if V is None or m is None:
        xArr = np.squeeze(np.array(X)).astype(np.float32)
        m, V = cv2.PCACompute(xArr)
        V = V[:100]

    for i in xrange(len(X)):
        # a = m[i].reshape(250, 250)
        # plt.imshow(a, cmap='gray')
        X[i] = np.dot(V, X[i]-m[0, i])
        # b = V[i].reshape(250, 250)
        # plt.imshow(b, cmap='gray')

    return X, V, m
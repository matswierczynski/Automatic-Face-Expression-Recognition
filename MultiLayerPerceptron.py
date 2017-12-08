import cv2
import numpy as np


class MLP:
    def __init__(self, layer_sizes, class_labels):

        self.no_of_features = layer_sizes[0]
        self.no_of_classes = layer_sizes[-1]
        self.class_labels = class_labels

        self.model = cv2.ANN_MLP()
        self.model.create(layer_sizes)

    def loadFromFile(self, f):
        self.model.load(f)

    def saveToFile(self, f):
        self.model.save(f)

    def prepareData(self, xTrain, yTrain):
        yTrain = self.labelsToIntArray(yTrain)
        yTrain = self.toOneHot(yTrain).reshape(-1, self.no_of_classes)

        self.model.train(xTrain, yTrain, None)

    def predict(self, xTest):
        _, yVote = self.model.predict(xTest)
        yVote = np.argmax(yVote, axis=1)
        return self.arrayOfIntsToString(yVote)

    def evaluate(self, xTest, yTest):
        yTest = self.labelsToIntArray(yTest)
        _, yVote = self.model.predict(xTest)
        accuracy = self.accuracy(yTest, yVote)

        return accuracy

    def labelsToIntArray(self, labels):
        arrayOfInts = np.zeros(len(labels)).astype(np.int32)
        for i in xrange (len(arrayOfInts)):
            index = 0
            for noOfLabels in xrange (self.no_of_classes):
                if labels[i] == self.class_labels[noOfLabels]:
                    index = noOfLabels
            arrayOfInts[i] = index
        return arrayOfInts

    def arrayOfIntsToString(self, intLabels):
        arrayOfStrings = np.zeros(len(intLabels)).astype(np.string_)
        for i in xrange(len(arrayOfStrings)):
            arrayOfStrings[i] = self.class_labels[intLabels[i]]
        return arrayOfStrings

    def toOneHot(self, y_train):
        numSamples = len(y_train)
        oneHot = np.zeros(numSamples*self.no_of_classes, np.float32)
        idx = np.int32(y_train + np.arange(numSamples)*self.no_of_classes)
        oneHot[idx] = 1
        return oneHot

    def accuracy(self, yTest, Yvote):
        maxVotes = np.argmax(Yvote, 1)
        good = maxVotes == yTest

        return np.count_nonzero(good)*1. / len(yTest)













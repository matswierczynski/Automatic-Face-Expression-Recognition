import numpy as np
from MultiLayerPerceptron import MLP
import LoadData


def main():
    (xTrain, yTrain), (xTest, yTest), V, m = LoadData.loadData(
        "datasets\\faces_training.pkl", splittingFactor=0.2, saveToFile="datasets\\faces_preprocessed.pkl")
    xTrain = np.squeeze(np.array(xTrain)).astype(np.float32)
    yTrain = np.squeeze(np.array(yTrain))
    xTest = np.squeeze(np.array(xTest)).astype(np.float32)
    yTest = np.squeeze(np.array(yTest))
    saveToFile = "datasets/mlp.xml"

    labels = np.unique(np.hstack((yTrain, yTest)))
    noOfFeatures = len(xTrain[0])
    noOfClasses = len(labels)

    bestAccuracy=0.0
    for i in xrange(10):
        layerSizes = np.int32([noOfFeatures, (i+1)*noOfFeatures/5, noOfClasses])
        MultiLayPer = MLP(layerSizes, labels)
        MultiLayPer.prepareData(xTrain, yTrain)
        acc = MultiLayPer.evaluate(xTest, yTest)
        print layerSizes, acc
        if acc > bestAccuracy:
            MultiLayPer.saveToFile(saveToFile)
            bestAccuracy = acc


if __name__ == '__main__':
    main()

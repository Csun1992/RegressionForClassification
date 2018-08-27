import numpy as np
import sys
from perceptron import * 

class Regression(Perceptron):
    def __init__(self, data, group):
        super(Regression, self).__init__(data, group)

    def getHatMatrix(self):
        pseudoInv = np.linalg.inv(self.data.T.dot(self.data)).dot(self.data.T)
        self.hatMatrix = self.data.dot(pseudoInv)
        return self.hatMatrix

    def classify(self):
        correctGroup = np.array(self.correctGroup).reshape(-1, 1)
        predictedVal = self.hatMatrix.dot(correctGroup)
        self.classifiedGroup = [sign(predictedVal[i]) for i in range(self.size)]
        return self.classifiedGroup
    
    def getMisclassified(self): 
        self.misclassified = [self.correctGroup[i] != self.classifiedGroup[i] for i in
        range(self.size)]
        return self.misclassified

    def getInSampleErr(self):
        return sum(self.misclassified)/float(self.size)

    def run(self):
        self.getHatMatrix()
        self.classify()
        self.getMisclassified()
        return self.getInSampleErr()



if __name__ == "__main__":
    experimentNum = 1000
    dataDim = 2
    lowerLim = -1
    upperLim = 1

    # experiment with sample size = 100
    sampleSize = 100
    inSampleErrs = []

    for i in range(experimentNum):
        data = np.random.uniform(lowerLim, upperLim, sampleSize*dataDim).reshape(sampleSize, dataDim)
        x1 = data[0, :]
        x2 = data[1, :]
        slope, intercept = findLine(x1, x2)
        group = classify(data, slope, intercept)

        regression = Regression(data, group)
        inSampleErrs.append(regression.run())
    print "For sample size = 100, the average in sample error is:"
    print sum(inSampleErrs)/experimentNum

    # estimate out-of-sample error
    hatMatrix = regression.getHatMatrix()
    data = np.random.uniform(lowerLim, upperLim, sampleSize*dataDim).reshape(sampleSize, dataDim)
    group = np.array(classify(data, slope, intercept)).reshape(-1, 1)
    predictedVals = hatMatrix.dot(group) 
    learnedGroup = [sign(predictedVals[i]) for i in range(sampleSize)]
    outSampleErr = sum([learnedGroup[i]!=group[i] for i in range(sampleSize)]) / float(sampleSize)

    print "out of sample error is:"
    print outSampleErr

    

import numpy as np
import sys
import perceptron as pp

def Regression(pp.Perceptron):
    def getHatMatrix():
        pseudoInv = np.linalg.inv(self.data.T.dot(self.data)) * self.data
        self.hatMatrix = self.data * pseudoInv
        return self.hatMatrix

    def classify():
        for i in range(self.size):
            self.classifiedGroup[i] = pp.sign(self.data[i, :].dot(self.hatMatrix))
        return self.classifiedGroup
    
    def getMisclassified(): 
        self.misclassified = [self.correctGroup[i] == self.classifiedGroup[i] for i in
        range(self.size)]
        return self.misclassified

    def getInSampleErr():
        return sum(self.misclassified)/float(self.size)

    def run():
        self.getHatMatrix()
        self.classify()
        self.getMisclassified()
        return self.getInSampleErr()



if __name__ == "__main__":
    experimentNum = 1000
    dataDim = 2
    lowerLim = -1
    upperLim = 1


    # experiment with sample size = 10
    sampleSize = 10
    repetition = []
    
    for i in range(experimentNum):
        data = np.random.uniform(lowerLim, upperLim, sampleSize*dataDim).reshape(sampleSize, dataDim)
        x1 = data[0, :]
        x2 = data[1, :]
        slope, intercept = findLine(x1, x2)
        group = classify(data, slope, intercept)

        perceptron = Perceptron(data, group)
        perceptron.learn()
        repetition.append(perceptron.getRepetition())

    repetition.sort()
    print "For sample size = 10, the average repetition is:"
    print sum(repetition)/experimentNum
    print repetition[len(repetition)/2]

    weight = perceptron.getCoeff()
    learnedSlope = -float(weight[1])/weight[2]
    learnedInter = -float(weight[0])/weight[2]

    testData = np.random.uniform(lowerLim, upperLim, dataDim*experimentNum).reshape(experimentNum, -1)
    testGroupClass = classify(testData, slope, intercept)
    learnedClass = classify(testData, learnedSlope, learnedInter) 
    errorRate = findErrorRate(testGroupClass, learnedClass)
    print "And the error rate is:"
    print errorRate

    
    # experiment with sample size = 100
    sampleSize = 100
    repetition = []

    for i in range(experimentNum):
        data = np.random.uniform(lowerLim, upperLim, sampleSize*dataDim).reshape(sampleSize, dataDim)
        x1 = data[0, :]
        x2 = data[1, :]
        slope, intercept = findLine(x1, x2)
        group = classify(data, slope, intercept)

        perceptron = Perceptron(data, group)
        perceptron.learn()
        repetition.append(perceptron.getRepetition())

    repetition.sort()
    print "For sample size = 100, the average repetition is:"
    print sum(repetition)/experimentNum
    print repetition[len(repetition)/2]

    weight = perceptron.getCoeff()
    learnedSlope = -float(weight[1])/weight[2]
    learnedInter = -float(weight[0])/weight[2]

    testData = np.random.uniform(lowerLim, upperLim, dataDim*experimentNum).reshape(experimentNum, -1)
    testGroupClass = classify(testData, slope, intercept)
    learnedClass = classify(testData, learnedSlope, learnedInter) 
    errorRate = findErrorRate(testGroupClass, learnedClass)
    print "And the error rate is:"
    print errorRate


import csv
import random
import math

class NaiveBayes:

    LAST_INDEX = -1

    def __init__(self):
        self.summaries = {}

    def loadCsv(self, filepath):
        lines = csv.reader(open(filepath, "r"))
        dataset = list(lines)

        for i in range(len(dataset)):
            dataset[i] = [float(x) for x in dataset[i]]

        return dataset

    def splitDataset(self, dataset, splitRatio):
        trainSize = int(len(dataset) * splitRatio)
        trainSet = []
        copy = list(dataset)

        while len(trainSet) < trainSize:
            index = random.randrange(len(copy))
            trainSet.append(copy.pop(index))

        return [trainSet, copy]

    def separateByClass(self, dataset):
        separated = {}

        for i in range(len(dataset)):
            vector = dataset[i]
            if vector[self.LAST_INDEX] not in separated:
                separated[vector[self.LAST_INDEX]] = []
            separated[vector[self.LAST_INDEX]].append(vector)

        return separated

    def mean(self, numbers):
        return sum(numbers) / float(len(numbers))

    def stdDev(self, numbers):
        avg = self.mean(numbers)
        variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
        return math.sqrt(variance)

    def summarize(self, dataset):
        summaries = [(self.mean(attribute), self.stdDev(attribute)) for attribute in zip(*dataset)]
        del summaries[-1]
        return summaries

    def summarizeByClass(self, dataset):
        separated = self.separateByClass(dataset)
        summaries = {}
        for classValue, instances in separated.items():
            summaries[classValue] = self.summarize(instances)
        return summaries

    # Gaussain Distribution Function
    def calculateProbability(self, x, mean, stddev):
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stddev, 2))))
        return (1 / (math.sqrt(2 * math.pi) * stddev)) * exponent

    def calculateClassProbability(self, summaries, inputVector):
        probabilities = {}
        for classValue, classSummaries in summaries.items():
            probabilities[classValue] = 1
            for i in range(len(classSummaries)):
                mean, stddev = classSummaries[i]
                x = inputVector[i]
                probabilities[classValue] *= self.calculateProbability(x, mean, stddev)
        return probabilities

    def predictWithSummaries(self, summaries, inputVector):
        probabilities = self.calculateClassProbability(summaries, inputVector)
        bestLabel, bestProb = None, -1
        for classValue, probability in probabilities.items():
            if bestLabel is None or probability > bestProb:
                bestProb = probability
                bestLabel = classValue
        return bestLabel

    def getPredictions(self, summaries, inputVector):
        predictions = []
        for i in range(len(inputVector)):
            result = self.predictWithSummaries(summaries, inputVector[i])
            predictions.append(result)
        return predictions

    def getAccuracy(self, testSet, predictions):
        correct = 0
        for x in range(len(testSet)):
            if testSet[x][self.LAST_INDEX] == predictions[x]:
                correct += 1
        return (correct/float(len(testSet))) * 100.0

    def fit(self, dataset):
        splitRatio = 0.80
        trainSet, testSet = self.splitDataset(dataset, splitRatio)
        self.summaries = self.summarizeByClass(trainSet)
        predictions = self.getPredictions(self.summaries, testSet)
        return self.getAccuracy(testSet, predictions)

    def predict(self, dataset):
        return self.getPredictions(self.summaries, dataset)


def main():
    filepath = 'data/pima-indians-diabetes.data'

    clf = NaiveBayes()
    dataset = clf.loadCsv(filepath)
    accuracy = clf.fit(dataset)
    print('Accuracy: {0}%'.format(accuracy))
    predictions = clf.predict(dataset[0:1])
    print('Prediction: {0}'.format(predictions))

main()

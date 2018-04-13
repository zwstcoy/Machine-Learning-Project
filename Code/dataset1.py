import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix as cf


#funtion for decision tree, get error
def DTtest(trainData, trainLabel, size):
    train_data, test_data, train_label, test_label = \
        train_test_split(trainData, trainLabel, test_size=size)

    dt = DecisionTreeClassifier()
    dt.fit(train_data, train_label)

    predict = dt.predict(test_data, test_label)

    # number of mix-matched label
    error = 0

    for x in range(len(predict)):
        if predict[x] != test_label[x]:
            error += 1

    # confusion matrix
    cf(test_label, predict)

    # accuracy
    print(1-(error / len(predict)))


    return error


def readFile(datafile):
    dataSet = []
    with open(datafile, 'r') as file:
        for line in file:
            data = line[:-1].split('\t')
            tmp = []
            for x in data:
                if x != '1.00000000000000e+99' and x != "":
                    x = float(x)
                    tmp.append(x)
                else:
                    tmp.append(-1)
            dataSet.append(tmp)
    return dataSet


# read files
trainDataFile = "Classification\TrainData1.txt"
trainLabelFile = "Classification\TrainLabel1.txt"
testDataFile = "Classification\TestData1.txt"

trainData = readFile(trainDataFile)
trainLabel = readFile(trainLabelFile)
testData = readFile(testDataFile)


# fill the missing value with mean
imr = Imputer(missing_values=-1, strategy='mean', axis=0)
imr = imr.fit(trainData)
trainData = imr.transform(trainData)

error = 0
length = 20
size = .15
for i in range(length):
    error += DTtest(trainData, trainLabel, size)


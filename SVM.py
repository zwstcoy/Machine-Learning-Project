from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.linear_model import Perceptron

#Data set 3 with 6300 * 13

def svmTest(trainData, trainLabel, size=.25):
    """
    :param trainData:2d array
    :param trainLabel: 1d array
    :param size: .2-.3
    :return: accuracy
    """
    train_data, test_data, train_label, test_label = \
        train_test_split(trainData, trainLabel, test_size=size)

    predict = svm(train_data, train_label, test_data)
    cf = confusion_matrix(test_label, predict)
    print("Confusion Matrix:")
    print(cf)
    accuracy = accuracy_score(test_label, predict)
    print(accuracy)
    return accuracy


def svm(train_data, train_label, test_data):
    cf = LinearSVC()
    cf.fit(train_data, train_label)
    return cf.predict(test_data)


def read_data_file(datafile, token):
    """
    :param datafile:
    :return: 2d array
    """
    dataset = []
    with open(datafile, 'r') as file:
        for line in file:
            #split each word by token
            data = line[:-1].split(token)
            tmp = []
            for x in data:
                if x != '' and x != '1.00000000000000e+99':
                    x = float(x)
                    tmp.append(x)
                else:
                    tmp.append(1e+99)
            dataset.append(tmp)
    return dataset


def read_label_file(datafile):
    """
    :param datafile:
    :return: a int array
    """
    dataset = []
    with open(datafile,'r') as file:
        for line in file:
            dataset.append(int(line))
    return dataset


# read files
trainDataFile = "Classification\TrainData1.txt"
trainLabelFile = "Classification\TrainLabel1.txt"
testDataFile = "Classification\TestData1.txt"

trainData = read_data_file(trainDataFile, '\t')
trainLabel = read_label_file(trainLabelFile)
testData = read_data_file(testDataFile, '\t')

label_tabel = [0]*10
# count number for each label
for x in trainLabel:
    label_tabel[int(x)-1] += 1
print("Number of each label")
print(label_tabel)

# fill the missing value with mean
imr = Imputer(missing_values=1e+99, strategy='mean', axis=0)
imr = imr.fit(trainData)
trainData = imr.transform(trainData)

# fill missing value in testdata with mean
imr = Imputer(missing_values=1e+99, strategy='mean', axis=0)
imr = imr.fit(testData)
testData = imr.transform(testData)


# use k-fold method to calculate average accuracy of DT
accuracy = 0
length = 10
size = .20
for i in range(length):
   accuracy += svmTest(trainData, trainLabel, size)

print("Average Accuracy",accuracy/length)

print(svm(trainData, trainLabel, testData))


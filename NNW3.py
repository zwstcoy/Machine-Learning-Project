from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import random
import numpy as np
import math


#Data set 3 with 6300 * 13

def NNWTest(trainData, trainLabel, size=.25):
    """
    :param trainData:2d array
    :param trainLabel: 1d array
    :param size: .2-.3
    :return: accuracy
    """
    train_data, test_data, train_label, test_label = \
        train_test_split(trainData, trainLabel, test_size=size)

    predict = NNW(train_data, train_label, test_data)
    cf = confusion_matrix(test_label, predict)
    print("Confusion Matrix:")
    print(cf)
    accuracy = accuracy_score(test_label, predict)
    print(accuracy)
    return accuracy


def NNW(train_data, train_label, test_data):
    nnw = MLPClassifier(activation ='logistic', solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(100, 100, 100),random_state=1)

    nnw.fit(train_data, train_label)
    return nnw.predict(test_data)

def removeDupulicate(data):
    remove = []
    distance = []
    length = len(data)
    for i in range(length):
        for j in range(i + 1, length):
            if i != j and j not in remove:
                dis = math.sqrt(sum([abs(data[i][x] - data[j][x]) for x in range(len(data[i]))]))
                distance.append(dis)
                if dis < 3:
                    remove.append(i)
    remove =set(remove)
    remove = np.array(sorted(remove, reverse= True))

    for x in remove:
        del data[x]
    return data

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
                    x = int(x)
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
trainDataFile = "Classification\TrainData3.txt"
trainLabelFile = "Classification\TrainLabel3.txt"
testDataFile = "Classification\TestData3.txt"

trainData = read_data_file(trainDataFile, '\t')
trainLabel = read_label_file(trainLabelFile)
testData = read_data_file(testDataFile, ',')

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
imr = Imputer(missing_values=1000000000, strategy='mean', axis=0)
imr = imr.fit(testData)
testData = imr.transform(testData)


#labelClass = [[],[],[],[],[],[],[],[],[]]
#for x in range(len(trainLabel)):
#    index = trainLabel[x]
#    labelClass[index-1].append(trainData[x])


#modify_traindata = []
#for i in range(len(labelClass)):
#    tmp = removeDupulicate(labelClass[i])
#    modify_traindata.append(tmp)
#    print(i+1, " ", len(modify_traindata[i]))


#modify_trainlabel = []
#for i in range(len(modify_traindata)):
#    for x in modify_traindata[i]:
#        modify_trainlabel.append(i+1)
#    print(i+1,"  ", modify_trainlabel.count(i+1))


#trainData = modify_traindata
#trainLabel = modify_trainlabel

# use k-fold method to calculate average accuracy of DT
accuracy = 0
length = 10
size = .10
for i in range(length):
    accuracy += NNWTest(trainData, trainLabel, size)

print("Average Accuracy",accuracy/length)

#print(NNW(trainData, trainLabel, testData))


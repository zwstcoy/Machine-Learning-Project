import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


trainDataFile = "Classification\TrainData1.txt"
trainLabelFile = "Classification\TrainLabel1.txt"
testDataFile = "Classification\TestData1.txt"

trainData = []
trainLabel = []
testData = []

count = 0

# read training dataset
with open(trainDataFile,'r') as file:
    for line in file:
        data = line[:-1].split('\t')
        tmp = []
        for x in data:
            if x!='1.00000000000000e+99' and x !=' ':
                x = float(x)
                tmp.append(x)
            else:
                tmp.append(-1)
                count +=1
        trainData.append(tmp)
print(count)

# read train label
with open(trainLabelFile,'r') as file:
    for line in file:
        data = line[:-1].split('\t')
        data = [int(x) for x in data]
        trainLabel.append(data)

# read test data
with open(testDataFile,'r') as file:
    for line in file:
        data = line[:-1].split('\t')
        tmp = []
        for x in data:
            if x!=" 1.00000000000000e+99" and  x !="":
                x = float(x)
                tmp.append(x)
            else:
                tmp.append(-1)
                count +=1

        testData.append(tmp)

print(testData)

# fill the missing value with mean
imr = Imputer(missing_values=-1, strategy='mean', axis=0)
imr = imr.fit(trainData)
trainData = imr.transform(trainData)

print(trainData)

train_data, test_data,  train_label,test_label = \
    train_test_split(trainData,trainLabel, test_size=.1)

print(len(train_data))
print(len(test_data))


dt = DecisionTreeClassifier()
dt.fit(train_data,train_label)

predict = dt.predict(test_data,test_label)

print(predict)
print(test_label)
error=0

for x in range(len(predict)):
    if predict[x]!=test_label[x]:
        print(x)
        print((predict[x]))
        print(test_label[x])

        error +=1

print(error/len(predict))

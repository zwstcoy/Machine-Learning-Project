import numpy as np
import matplotlib as plt

trainDataFile = "C:\Study\Python_WorkSpace\MLProject\Classification\TrainData1.txt"
trainLabelFile = "C:\Study\Python_WorkSpace\MLProject\Classification\TrainLabel1.txt"
testDataFile = "C:\Study\Python_WorkSpace\MLProject\Classification\TestData1.txt"

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
        trainData.append(data)
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

print(count)
trainData = np.array(trainData)
trainLabel = np.array(trainLabel)
testData = np.array(testData)


print(trainData)





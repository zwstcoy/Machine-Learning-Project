from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from scipy.stats import mode
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


def Test(trainData, trainLabel, size=.25):
    """
    :param trainData:2d array
    :param trainLabel: 1d array
    :param size: .2-.3
    :return: accuracy
    """
    train_data, test_data, train_label, test_label = \
        train_test_split(trainData, trainLabel, test_size=size)

    predictNNW = NNW(train_data, train_label, test_data)
    predictDT = decisionTree(train_data, train_label, test_data)
    predictLR = logreg(train_data, train_label, test_data)

    accuracyNNW = accuracy_score(test_label, predictNNW)
    accuracyDT = accuracy_score(test_label, predictDT)
    accuracyLR = accuracy_score(test_label, predictLR)
    accuracy = (accuracyNNW+accuracyDT+accuracyLR)/3
    print(accuracy)
    return accuracy

def NNW(train_data, train_label, test_data):
    nnw = MLPClassifier(activation ='logistic', solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(100, 100, 100),random_state=1)

    nnw.fit(train_data, train_label)
    return nnw.predict(test_data)

def decisionTree(train_data, train_label, test_data):
    # balance the data
    params = {'n_estimators': 100, 'random_state': 0,
              'class_weight': 'balanced', 'min_samples_split': .2 }

    # one type of decision tree that use for imbalance data set
    dt = RandomForestClassifier(**params)
    dt.fit(train_data, train_label)
    predict = dt.predict(test_data)
    return predict

def logreg(train_data, train_label, test_data):

    # one type of decision tree that use for imbalance data set
    log = LogisticRegression(solver='saga', class_weight='balanced', multi_class='multinomial')

    log.fit(train_data, train_label)
    predict = log.predict(test_data)
    return predict

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
                if x != "":
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
trainDataFile = "Classification/TrainData3.txt"
trainLabelFile = "Classification/TrainLabel3.txt"
testDataFile = "Classification/TestData3.txt"

trainData = read_data_file(trainDataFile, '\t')
trainLabel = read_label_file(trainLabelFile)
testData = read_data_file(testDataFile, ',')

# fill the missing value with mean
imr = Imputer(missing_values=1e+99, strategy='mean', axis=0)
imr = imr.fit(trainData)
trainData = imr.transform(trainData)

# fill missing value in testdata with mean
imr = Imputer(missing_values=1000000000, strategy='mean', axis=0)
imr = imr.fit(testData)
testData = imr.transform(testData)


# use k-fold method to calculate average accuracy of DT
accuracy = 0
length = 10
size = .20
#for i in range(length):
  # accuracy += Test(trainData, trainLabel, size)
#print("Average Accuracy",accuracy/length)



# DT FINAL TEST RESULT
dt_1 = decisionTree(trainData,trainLabel,testData)
dt_2 = decisionTree(trainData,trainLabel,testData)
dt_3 = decisionTree(trainData,trainLabel,testData)
dt_4 = decisionTree(trainData,trainLabel,testData)
dt_5 = decisionTree(trainData,trainLabel,testData)
#print(final_dt)

# NNW FINAL TEST RESULT
nnw_1 = NNW(trainData, trainLabel, testData)
nnw_2 = NNW(trainData, trainLabel, testData)
nnw_3 = NNW(trainData, trainLabel, testData)
nnw_4 = NNW(trainData, trainLabel, testData)
nnw_5 = NNW(trainData, trainLabel, testData)
#print(final_nnw)

# log Reg FINAL TEST RESULT

log_1 = logreg(trainData, trainLabel, testData)
log_2 = logreg(trainData, trainLabel, testData)
log_3 = logreg(trainData, trainLabel, testData)
log_4 = logreg(trainData, trainLabel, testData)
log_5 = logreg(trainData, trainLabel, testData)
#print(final_log)


m = list(mode([dt_1, dt_2, dt_3, dt_4, dt_5, nnw_1, nnw_2, nnw_3, nnw_4, nnw_5, log_1,
               log_2, log_3, log_4, log_5]))
print(m[0][0])

result_file = open('Zheng_Ho_Classification13.txt', 'w')
for item in m[0][0]:
    print(item)
    result_file.write("%s \t" % item)

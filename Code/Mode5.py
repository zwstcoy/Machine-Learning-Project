from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from scipy.stats import mode

# LOG REG
def logreg(train_data, train_label, test_data):
    log = LogisticRegression(solver='newton-cg', class_weight='balanced', multi_class='multinomial', max_iter=1000)
    log.fit(train_data, train_label)
    return log.predict(test_data)

# NNW
def NNW(train_data, train_label, test_data):
    nnw = MLPClassifier(activation ='logistic', solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(100, 100, 100,100),random_state=1)

    nnw.fit(train_data, train_label)
    return nnw.predict(test_data)

# DT
def decisionTree(train_data, train_label, test_data):
    # balance the data
    params = {'n_estimators': 500, 'random_state': 0,
              'class_weight': 'balanced', 'min_samples_split': .2 }

    # one type of decision tree that use for imbalance data set
    dt = RandomForestClassifier(**params)
    dt.fit(train_data, train_label)
    predict = dt.predict(test_data)
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
                if x != '' and x != '1.00000000000000e+99':
                    x = float(x)
                    tmp.append(x)
                elif x == '1.00000000000000e+99':
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
            if line !='\n':
                dataset.append(int(line))
    return dataset


# read files
trainDataFile = "Classification/TrainData5.txt"
trainLabelFile = "Classification/TrainLabel5.txt"
testDataFile = "Classification/TestData5.txt"

trainData = read_data_file(trainDataFile, ' ')
trainLabel = read_label_file(trainLabelFile)
testData = read_data_file(testDataFile, ' ')


# fill the missing value with mean value in the column
imr = Imputer(missing_values=1e+99, strategy='mean', axis=0)
imr = imr.fit(trainData)
trainData = imr.transform(trainData)

# fill missing value in testdata with mean value in the column
imr = Imputer(missing_values=1e+99, strategy='mean', axis=0)
imr = imr.fit(testData)
testData = imr.transform(testData)

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
print(m[0])

result_file = open('Zheng_Ho_Classification15.txt', 'w')

for item in m[0][0]:
    result_file.write("%s \t" % item)

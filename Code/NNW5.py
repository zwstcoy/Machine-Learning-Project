from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from scipy.stats import mode
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

#Work slow

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


def logreg(train_data, train_label, test_data):
    log = LogisticRegression(solver='newton-cg', class_weight='balanced', multi_class='multinomial', max_iter=500)
    log.fit(train_data, train_label)
    return list(log.predict(test_data))

# NNW
def NNW(train_data, train_label, test_data):
    nnw = MLPClassifier(activation ='logistic', solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(100, 100, 100,100),random_state=1)

    nnw.fit(train_data, train_label)
    return list(nnw.predict(test_data))

# DT
def decisionTree(train_data, train_label, test_data):
    # balance the data
    params = {'n_estimators': 100, 'random_state': 0,
              'class_weight': 'balanced', 'min_samples_split': .2 }

    # one type of decision tree that use for imbalance data set
    dt = RandomForestClassifier(**params)
    dt.fit(train_data, train_label)
    predict = dt.predict(test_data)
    return list(predict)


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
print(trainData[:1])

label_tabel = [0]*10
# count number for each label
for x in trainLabel:
    label_tabel[int(x)-1] += 1
print("Number of each label")
print(label_tabel)

# fill the missing value with mean value in the column
imr = Imputer(missing_values=1e+99, strategy='mean', axis=0)
imr = imr.fit(trainData)
trainData = imr.transform(trainData)

# fill missing value in testdata with mean value in the column
imr = Imputer(missing_values=1e+99, strategy='mean', axis=0)
imr = imr.fit(testData)
testData = imr.transform(testData)

# use k-fold method to calculate average accuracy of DT
accuracy = 0
length = 10
size = .30
for i in range(length):
   accuracy += NNWTest(trainData, trainLabel, size)

print("Average Accuracy",accuracy/length)

#final_test_data_predict = NNW(trainData, trainLabel, testData)
#print([x for x in final_test_data_predict])

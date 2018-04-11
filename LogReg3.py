from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# function for decision tree, get error
def DTtest(trainData, trainLabel, size):
    """
    :param trainData:2d array
    :param trainLabel: 1d array
    :param size: .2-.3
    :return: accuracy
    """
    train_data, test_data, train_label, test_label = \
        train_test_split(trainData, trainLabel, test_size=size)

    predict = logreg(train_data, train_label, test_data)

    cf = confusion_matrix(test_label,predict)
    print("Confusion Matrix:")
    print(cf)
    accuracy = accuracy_score(test_label, predict)
    print(accuracy)
    return accuracy


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


# use k-fold method to calculate average accuracy of DT
accuracy = 0
length = 10
size = .20
for i in range(length):
   accuracy += DTtest(trainData, trainLabel, size)

print("Average Accuracy",accuracy/length)

# this is the decision tree method run on final test dataset, the final_test_data_predict is the predict label for test dataset
final_test_data_predict = logreg(trainData,trainLabel,testData)

print([x for x in final_test_data_predict])
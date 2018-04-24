import math


def readFile(data_file):
    """
    :param data_file: text file
    :return: row array, and missing row array with the row have missing value
    """
    row = []
    with open(data_file, 'r') as file:
        for line in file:
            tmp = line.split("\t")
            row.append([float(x) for x in tmp])
    return row


def find_missing_col(missing_row):
    """
    :param missing_row: the 2d array contain row that have at least one missing
    :return: the missing columon in each row
    """
    missing_col = []
    for x in missing_row:
        tmp = []
        for i, t in enumerate(x):
            if t == 1e+99:
                tmp.append(i)
        missing_col.append(tmp)
    return missing_col


def calculate_row_dis(row, missing_row, col):
    """
    :param row: row that don't contain missing value
    :param missing_row: row have at lease one missing value
    :param col: missing value columns in the row
    :return:
    """
    distance = []
    for x in row:
        row_dis = 0
        for i, y in enumerate(x):
            if i not in col and y != 1e+99:
                row_dis += ((y - missing_row[i])*(y - missing_row[i]))
        distance.append(math.sqrt(row_dis))
    return distance


def cal_missing_val(row, top_k_dis_list, missing_col):
    """
    :param row: 2d array with missing value
    :param top_k_dis_list: the top k close distance row index
    :param missing_col: the array contain missing value
    :return:
    """
    col_precent = []
    dem = 1
    for x in top_k_dis_list:
        if row[x][missing_col] != 1e+99:
            dem += row[x][missing_col]
    for x in top_k_dis_list:
        col_precent.append((math.fabs(row[x][missing_col])) / dem)
    col_val = []
    for i, x in enumerate(top_k_dis_list):
        col_val.append(float(row[x][missing_col]) * col_precent[i])
    return sum(col_val)


data_file = "MissingValueEstimation/MissingData3.txt"

row = readFile(data_file)
missing_col = find_missing_col(row)

print("Number of missing value row: ", len(row))
print("Total number of the misssing value: ", sum(len(x) for x in missing_col))

total_distance = []
top_k_dis_list = []
for x in range(len(row)):
    total_distance.append(calculate_row_dis(row, row[x], missing_col[x]))


for rows in total_distance:
    top_k = []
    for x in range(9):
        minIndex = 0
        for index in range(len(rows)):
            if rows[index] < rows[minIndex] and index not in top_k and row[index]!=1e+99:
                minIndex = index
        top_k.append(minIndex)
    top_k_dis_list.append(top_k)
print(top_k_dis_list[0])

missing_val = []

for lines in missing_col:
    val = []
    for x in range(len(lines)):
         val.append(cal_missing_val(row, top_k_dis_list[x], lines[x]))
    missing_val.append(val)
print(missing_val)
# fill the missing with the predict value
for index, lines in enumerate(missing_col):
    for i, x in enumerate(lines):
        row[index][x] = missing_val[index][i]


result_file3 = open('Zheng_Ho_MissingResult3.txt', 'w')

for item in row:
    result_file3.write("%s\n" % item)
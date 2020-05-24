import numpy as np
from urllib.request import urlopen
from sklearn import datasets, linear_model
from math import sqrt
import matplotlib.pyplot as plt

def select_attributes(attributes, selection):
    subset = []
    for record in attributes:
        subset.append([record[column] for column in selection])
    return subset

host_url = ("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv")
dataset = urlopen(host_url)

attributes = []; labels = []; names = []
FIRST_LOOP_FLAG = True
for row in dataset:
    if FIRST_LOOP_FLAG:
        # collect attribute names
        names = row.strip().decode().split(';')
        FIRST_LOOP_FLAG = False
    else:
        record = row.strip().decode().split(';')
        labels.append(float(record[-1]))
        record.pop()
        record = [float(num) for num in record]
        attributes.append(record)

# split data into test and train sets
attributes_test = [attributes[i] for i in range(len(attributes)) if (i % 3 == 0)]
attributes_train = [attributes[i] for i in range(len(attributes)) if (i % 3 != 0)]
labels_test = [labels[i] for i in range(len(labels)) if (i % 3 == 0)]
labels_train = [labels[i] for i in range(len(labels)) if (i % 3 != 0)]

# build 'best' attribute list
attribute_list = []
record_set = set(range(len(attributes[1])))

for record in record_set:
    attribute_set = set(attributes)
    test_set = list(record_set - attribute_set)
    errors = []; temp = []
    # try each new attribute and record performance of trained model
    for test in test_set:
        temp += attribute_list
        temp.append(test)
        # form training and testing sub matrices
        x_train = select_attributes(attributes_train, temp)
        x_test = select_attributes(attributes_test, temp)
        # convert to numpy array for performance
        x_train = np.array(x_train)
        y_train = np.array(labels_train)
        x_test = np.array(x_test)
        y_test = np.array(labels_test)

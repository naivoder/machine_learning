from urllib.request import urlopen
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import roc_curve, auc
import pylab as pl

host_url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data")
dataset = urlopen(host_url)

# data structure for performance prediction
def confusion_matrix(predicted, actual, threshold, debug=False):
    if len(predicted) != len(actual): return -1
    # true pos, false pos, true neg, false neg
    tp, fp, tn, fn = [0.0] * 4
    if debug:
        print(tp, fp, tn, fn)
    for index in range(len(actual)):
        # positive example
        if debug:
            print(predicted[index], actual[index])
        if actual[index] > 0.5:
            # positive prediction
            if predicted[index] > threshold:
                tp += 1.0
            # negative prediction
            else:
                fn += 1.0
        # negative example
        else:
            # positive prediction
            if predicted[index] > threshold:
                fp += 1.0
            # negative prediction
            else:
                tn += 1.0
    return [tp, fn, fp, tn]

# split labels into separate lists
attributes = []; labels = []
for row in dataset:
    record = row.strip().decode().split(',')
    # Mine = 1, Rock = 0
    if (record[-1] == 'M'):
        labels.append(1.0)
    else:
        labels.append(0.0)
    # remove label from attributes matrix
    record.pop()
    # convert ints to floats
    record = [float(num) for num in record]
    attributes.append(record)

# split dataset into test/train @ 2/3
# pandas_analysis shows non-homogeneous data, must skip indices
attribute_test = [attributes[i] for i in range(len(attributes)) if (i % 3 == 0)]
attribute_train = [attributes[i] for i in range(len(attributes)) if (i % 3 != 0)]
labels_test = [labels[i] for i in range(len(labels)) if (i % 3 == 0)]
labels_train = [labels[i] for i in range(len(labels)) if (i % 3 != 0)]

# convert to numpy arrrays
x_train = np.array(attribute_train); y_train = np.array(labels_train)
x_test = np.array(attribute_test); y_test = np.array(labels_test)

# confirm that everything is as expected...
print("--------")
print("Training")
print("--------")
print("Shape of attributes:", x_train.shape)
print("Shape of labels:", y_train.shape)
print("-------")
print("Testing")
print("-------")
print("Shape of attributes:", x_test.shape)
print("Shape of labels:", y_test.shape)

# linear regression model
rocks_v_mines = linear_model.LinearRegression()
rocks_v_mines.fit(x_train, y_train)

# generate error predictions
training_predictions = rocks_v_mines.predict(x_train)
print("Sample predictions:", training_predictions[0:3], training_predictions[-4:-1])

# generate and display confusion matrix data (set desired threshold here)
tp, fn, fp, tn  = confusion_matrix(training_predictions, y_train, 0.5)
print("TP: %s\tFN: %s\t\tFP: %s\t\tTN: %s" % (tp, fn, fp, tn))

"""
this file implements forward stepwise regression to determine the 'complexity parameter', or number of attributes to be used in the model
the method also provides a numerical importance weight for each attribute, and a means for selecting the best group of attributes for all available
the selection algorithm works by starting with an empty 'best' list and adding the attribute that scores highest when used to train the model alone
with each successive pass the available attributes are tested one at a time with the 'best' set and the highest scoring attribute is kept

"""

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
attribute_list = []; oos_error = []
record_set = set(range(len(attributes[1])))

for record in record_set:
    # list is empty on first pass
    attribute_set = set(attribute_list)
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

        # train model with set of attributes
        wine_quality = linear_model.LinearRegression()
        wine_quality.fit(x_train, y_train)
        # generate predictions and calculate rms error
        rms_error = np.linalg.norm((y_test-wine_quality.predict(x_test)), 2)/sqrt(len(y_test))
        errors.append(rms_error)
        temp = []

    optimal = np.argmin(errors)
    attribute_list.append(test_set[optimal])
    oos_error.append(errors[optimal])

names = [names[attribute] for attribute in attribute_list]
best_attributes = zip(attribute_list, names)

print("Out of Sample Error vs. Attribute Set Size")
print(oos_error)
print("Best Attributes")
for item in best_attributes:
    print(item)

points = range(len(oos_error))
plt.plot(points, oos_error, 'k')
plt.xlabel("Number of Attributes")
plt.ylabel("RMS Error")
plt.show()

best = oos_error.index(min(oos_error))
best_set = attribute_list[1:(best+1)]

x_train = np.array(select_attributes(attributes_train, best_set))
x_test = np.array(select_attributes(attributes_test, best_set))

wine_quality = linear_model.LinearRegression()
wine_quality.fit(x_train, y_train)
error = y_test - wine_quality.predict(x_test)
plt.hist(error)
plt.xlabel("Bin Boundaries")
plt.ylabel("Counts")
plt.show()

plt.scatter(wine_quality.predict(x_test), y_test, s=100, alpha=0.1)
plt.xlabel("Predicted Score")
plt.ylabel("Actual Score")
plt.show()

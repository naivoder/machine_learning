"""
this file uses pandas to produce a parallel coordinate plot of the wine tasting data. the results show high scores aggregating at high alcohol values and low scores aggregating at high values of volatile acidity 

"""
import math
import pandas as pd
import matplotlib.pyplot as plt

host_url = ("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv")
dataset = pd.read_csv(host_url, header=0, sep=';')

summary = dataset.describe()
number_of_rows = len(dataset.index)
mean_taste = summary.iloc[1, -1]
std_taste = summary.iloc[2, -1]
number_of_attributes = len(dataset.columns) - 1

for i in range(number_of_rows):
    record = dataset.iloc[i, 0:number_of_attributes]
    normalized_target = (dataset.iloc[i, number_of_attributes] - mean_taste) / std_taste
    # increment of datapoint from baseline, as % of total range
    label_color = 1.0 / (1.0 + math.exp(-normalized_target))
    record.plot(color=plt.cm.RdYlBu(label_color), alpha=0.5)

plt.xlabel("Attribute Index")
plt.ylabel("Attribute Values")
plt.show()

normalized_data = dataset
number_of_columns = len(normalized_data.columns)

for i in range(number_of_columns):
    mean = summary.iloc[1, i]
    std = summary.iloc[2, i]
    normalized_data.iloc[:, i:(i + 1)] = (normalized_data.iloc[:, i:(i + 1)] - mean) / std

for i in range(number_of_rows):
    record = normalized_data.iloc[i, 0:number_of_attributes]
    normalized_target = normalized_data.iloc[i, number_of_attributes]
    label_color = 1.0 / (1.0 + math.exp(-normalized_target))
    record.plot(color=plt.cm.RdYlBu(label_color), alpha=0.5)

plt.xlabel("Attribute Index")
plt.ylabel("Attribute Values")
plt.show()

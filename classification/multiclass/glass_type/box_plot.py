"""
this file uses pandas to generate a boxplot of the attribute interquartile ranges
the box plot shows an unusually high number of outliers, due to attribute values not necessarily having any continuity in relationship with class value, i.e. no reason to expect proximity of attribute values across classes
the data is also unbalanced, there are large descrepancies in number of examples provided for different classes

"""
import pylab
import pandas as pd
import matplotlib.pyplot as plt

host_url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data")
dataset = pd.read_csv(host_url, header=None)
column_count = len(dataset.columns)

# first column is i.d. variable -> strip it off
normalized = dataset.iloc[:, 1:column_count]
norm_column_count = len(normalized.columns)
# get summary dataframe
summary = normalized.describe()

# calculate mean and std. deviation
for i in range(norm_column_count):
    mean = summary.iloc[1, i]
    sd = summary.iloc[2, i]
    # normalize data
    normalized.iloc[:, i:(i+1)] = (normalized.iloc[:, i:(i+1)] - mean) / sd
norm_data = normalized.values

# plot data
pylab.boxplot(norm_data)
plt.xlabel("Attribute Index")
plt.ylabel("Quartile Ranges - Normalized")
pylab.show()

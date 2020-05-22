"""
this file uses pandas to produce a pc plot with 7 distinct color groups to help visualize the relationship between different attributes and the target labels

"""


import pandas as pd
import matplotlib.pyplot as plot

host_url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data")
dataset = pd.read_csv(host_url, header=None)
dataset.columns = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']

normalized = dataset
row_count = len(normalized.index)
column_count = len(normalized.columns)
attribute_count = column_count - 1
summary = normalized.describe()

# normalize data (!labels)
for i in range(attribute_count):
    mean = summary.iloc[1, i]
    sd = summary.iloc[2, i]
    normalized.iloc[:, i:(i+1)] = (normalized.iloc[:, i:(i+1)] - mean) / sd

for i in range(row_count):
    # ignore Id column
    record = normalized.iloc[i, 1:attribute_count]
    # 7 possible labels
    label_color = normalized.iloc[i, attribute_count] / 7.0
    record.plot(color=plot.cm.RdYlBu(label_color), alpha=0.5)

plot.xlabel("Attribute Index")
plot.ylabel("Attribute Value")
plot.show()

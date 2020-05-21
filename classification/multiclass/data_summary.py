"""
this file uses pandas to give a summary of the attributes, their types, their value ranges and their general placement within the data structure
the summary shows 214 records with 10 attributes and a numeric label corresponding to quality. the data appears to be separated distinctly by type.

"""
import pandas as pd
import matplotlib.pyplot as plt

host_url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data")
dataset = pd.read_csv(host_url, header=None)

# print head and tail
print("Head:")
print(dataset.head())
print("Tail:")
print(dataset.tail())

# print summary
summary = dataset.describe()
print("Summary:")
print(summary)

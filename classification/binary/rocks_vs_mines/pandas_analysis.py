"""
this file uses pandas to read and summarize the given dataset

"""
import pandas as pd

host_url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data")

dataset = pd.read_csv(host_url, header=None, prefix="C-")

# print head and tail
print("Head:")
print(dataset.head())
print("Tail:")
print(dataset.tail())

# print summary
summary = dataset.describe()
print("Summary:")
print(summary)

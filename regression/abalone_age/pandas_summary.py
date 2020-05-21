"""
this file uses pandas to summarize and gain insight into the abalone dataset for the purposes of generating a machine learning algorithm that can predict the age based on given data
the head/tail tells that the data has eight attributes and a corresponding numeric label for each record. The data is not separated by sex (the first attribute column is categorical) and there are 4176 total records in the dataset.
the summary provides a dataframe with statistical information for each attribute

"""
import pylab
import pandas as pd
import matplotlib.pyplot as plt

host_url = ("http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data")
dataset = pd.read_csv(host_url, header=None)
dataset.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole Weight', 'Shucked Weight', 'Viscera Weight', 'Shell Weight', 'Rings']

# print head and tail
print("Head:")
print(dataset.head())
print("Tail:")
print(dataset.tail())

# print summary
summary = dataset.describe()
print("Summary:")
print(summary)

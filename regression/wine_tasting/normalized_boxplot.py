"""
this file uses pandas to create a normalized boxplot, giving a visual representation for the interquartile ranges of the various attribute columns
this boxplot confirms the general numeric summary, i.e. there are numerous outlying values that could potentially be a source of error when analyzing the performance of the predictive model 

"""

from pylab import *
import pandas as pd
import matplotlib.pyplot as plt

host_url = ("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv")
dataset = pd.read_csv(host_url, header=0, sep=';')

summary = dataset.describe()

normal_data = dataset
for i in range(len(normal_data.columns)):
    mean = summary.iloc[1, i]
    sd = summary.iloc[2, i]
    normal_data.iloc[:, i:(i + 1)] = (normal_data.iloc[:, i:(i + 1)] - mean) / sd

normalized = normal_data.values
boxplot(normalized)
plt.xlabel("Attribute Index")
plt.ylabel("Quartile Ranges - Normalized")
show()

"""
this file uses pandas to create a correlation matrix and heatmap between the attributes (and the targets, since predictions are numeric)

"""

import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

host_url = ("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv")
dataset = pd.read_csv(host_url, header=0, sep=';')

# omit non-numeric sex attribute
correlation_matrix = DataFrame(dataset.corr())
print(correlation_matrix)

plt.pcolor(correlation_matrix, cmap='inferno')
plt.show()

"""
this file uses pandas to produce a correlation heat map from the data, less effective in this instance because the targets take on one of several discrete values and must therefor be left off

"""

import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plot

host_url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data")
dataset = pd.read_csv(host_url, header=None)
dataset.columns = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']

attribute_count = len(dataset.columns) - 1
# ignore Id column and label column
correlation_matrix = DataFrame(dataset.iloc[:, 1:-1].corr())

plot.pcolor(correlation_matrix)
plot.show()

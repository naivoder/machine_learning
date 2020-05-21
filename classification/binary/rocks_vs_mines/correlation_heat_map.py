"""
this file uses pandas to create a correlation heat map of all the attributes in a given dataset

"""
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

host_url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data")
dataset = pd.read_csv(host_url, header=None)

correlation_matrix = DataFrame(dataset.corr())

plt.pcolor(correlation_matrix)
plt.show()

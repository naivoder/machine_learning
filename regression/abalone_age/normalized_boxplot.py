from pylab import *
import pandas as pd
import matplotlib.pyplot as plt

host_url = ("http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data")
dataset = pd.read_csv(host_url, header=None)
dataset.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole Weight', 'Shucked Weight', 'Viscera Weight', 'Shell Weight', 'Rings']

real_values = dataset.iloc[:,1:9].values
boxplot(real_values)
plt.xlabel("Attribute Index")
plt.ylabel("Quartile Ranges")
show()

minus_outlier = dataset.iloc[:,1:8].values
boxplot(minus_outlier)
plt.xlabel("Attribute Index")
plt.ylabel("Quartile Ranges")
show()

summary = dataset.describe()

normal_data = dataset.iloc[:,1:9]
for i in range(8):
    mean = summary.iloc[1, i]
    sd = summary.iloc[2, i]
    normal_data.iloc[:, i:(i + 1)] = (normal_data.iloc[:, i:(i + 1)] - mean) / sd

normalized = normal_data.values
boxplot(normalized)
plt.xlabel("Attribute Index")
plt.ylabel("Quartile Ranges")
show()

"""
this file uses pandas to construct a parallel coordinate plot from the given dataset, which can help visualize systematic relationships between attribute values and labels

"""
import pandas as pd
import matplotlib.pyplot as plt

host_url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data")

dataset = pd.read_csv(host_url, header=None, prefix="C-")

# known value for number of records
number_of_rows = dataset.shape[0]
for index in range(number_of_rows):
    if dataset.iat[index, 60] == 'M':
        label_color = 'green'
        alpha = 0.6

    else:
        label_color = 'blue'
        alpha = 1.0

    record = dataset.iloc[index, 0:60]
    record.plot(color=label_color, alpha=alpha)

plt.xlabel("Attribute Index")
plt.ylabel("Attribute Values")
plt.show()

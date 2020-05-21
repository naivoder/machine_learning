"""
this file uses pandas to create a scatter plot showing the correlation between attribute 35 and the target labels.
attribute 35 was chosen based on the visible separation of the categories in the parallel_coordinates_plot.

"""

import pandas as pd
import matplotlib.pyplot as plt
from random import uniform

host_url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data")

dataset = pd.read_csv(host_url, header=None, prefix="C-")

target = []
number_of_rows = dataset.shape[0]
for index in range(number_of_rows):
    if dataset.iat[index, 60] == 'M':
        target.append(1.0 + uniform(-0.1, 0.1))
    else:
        target.append(0.0 + uniform(-0.1, 0.1))

attribute_35 = dataset.iloc[:,35]
plt.scatter(attribute_35, target, alpha=0.5, s=120)

plt.xlabel("Attribute")
plt.ylabel("Target")
plt.show()

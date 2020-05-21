"""
this file uses pandas to create a scatter (cross) plot, to help visualize the relationships between various attributes

"""
import random
import pandas as pd
import matplotlib.pyplot as plt

host_url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data")

dataset = pd.read_csv(host_url, header=None, prefix="C-")

# correlation between selected or random attributes
location_a = random.randint(0, dataset.shape[0])
location_b = random.randint(0, dataset.shape[0])
record_a = dataset.iloc[location_a, 0:60]
record_b = dataset.iloc[location_b, 0:60]

plt.scatter(record_a, record_b)
plt.xlabel("Attribute A (%s)" % location_a)
plt.ylabel("Attribute B (%s)" % location_b)
plt.show()

record_a = dataset.iloc[1, 0:60]
record_b = dataset.iloc[2, 0:60]

plt.scatter(record_a, record_b)
plt.xlabel("Attribute 2")
plt.ylabel("Attribute 3")
plt.show()

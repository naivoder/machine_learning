"""
this file uses pandas to create a parallel coordinates plot for the abalone regression problem assigning different color shades to higher and lower target values. the second plot normalizes the data. this plot can take a little while to render...
this plot indicates significant correlation between each of the available measurable attributes and abalone age (number of rings). since similar shades are largely grouped together at similar values an accurate predictive model should be possible.
the areas of overlap indicate examples that will be difficult to predict correlectly from any single attribute. 

"""
import math
import pandas as pd
import matplotlib.pyplot as plt

host_url = ("http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data")
dataset = pd.read_csv(host_url, header=None)
dataset.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole Weight', 'Shucked Weight', 'Viscera Weight', 'Shell Weight', 'Rings']

summary = dataset.describe()
min_rings = summary.iloc[3, 7]
max_rings = summary.iloc[7, 7]
number_of_rows = len(dataset.index)

for i in range(number_of_rows):
    record = dataset.iloc[i, 1:8]
    # increment of datapoint from baseline, as % of total range
    label_color = (dataset.iloc[i, 8] - min_rings) / (max_rings - min_rings)
    record.plot(color=plt.cm.RdYlBu(label_color), alpha=0.5)

plt.xlabel("Attribute Index")
plt.ylabel("Attribute Values")
plt.show()

mean_rings = summary.iloc[1, 7]
std_rings = summary.iloc[2, 7]

for i in range(number_of_rows):
    record = dataset.iloc[i, 1:8]
    normalized_record = (dataset.iloc[i, 8] - mean_rings) / std_rings
    label_color = 1.0 / (1.0 + math.exp(-normalized_record))
    record.plot(color=plt.cm.RdYlBu(label_color), alpha=0.5)

plt.xlabel("Attribute Index")
plt.ylabel("Attribute Values")
plt.show()

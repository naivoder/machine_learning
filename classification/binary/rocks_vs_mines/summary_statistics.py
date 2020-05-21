"""
this file outputs descriptive statistics for each (desired) numeric attribute column and a count of unique categories in each categorical attribute

"""

from urllib.request import urlopen
import numpy as np

host_url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data")
dataset = urlopen(host_url)

attributes = []; labels = []
for record in dataset:
    row = record.strip().decode().split(',')
    attributes.append(row)

number_of_rows = len(attributes)
number_of_columns = len(attributes[1])

column_stats = []

for column in range(number_of_columns-1):
    column_data = []
    for row in attributes:
        try:
            feature = float(row[column])
            if isinstance(feature, float):
                column_data.append(float(row[column]))
        except ValueError:
            continue
    column_array = np.array(column_data)
    column_array = column_array
    column_mean = np.mean(column_array)
    column_std_deviation = np.std(column_array)
    bounds = np.quantile(column_array, [0.25,0.5,0.75])
    column_stats.append([column_mean, column_std_deviation, bounds])

print("*___Numeric Attributes___*")
print("Column\tMean\t\tStd. Deviation\tQ(.25)\tQ(.50)\tQ(.75)")
index = 0
for stats in column_stats:
    print("%s\t%6.5f\t\t%6.5f\t\t%6.5f\t%6.5f\t%6.5f" % (index, stats[0], stats[1], stats[2][0], stats[2][1], stats[2][2]))
    index += 1
print("\n*___Categorical Attributes___*")
column = 60; category_data = []
for row in attributes:
    category_data.append(row[column])
unique = set(category_data)
labels = dict(zip(list(unique), len(unique)*[0]))

for label in category_data:
    labels[label] += 1

print("Unique labels:", labels)

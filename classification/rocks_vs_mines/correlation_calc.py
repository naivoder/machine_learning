"""
this file implements Pearson's coorelation coefficient to calculate correlation for the variables analyzed in scatter_plot.py

"""
import random
import pandas as pd
from math import sqrt

host_url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data")

dataset = pd.read_csv(host_url, header=None, prefix="C-")

# correlation between selected or random attributes
record_a = dataset.iloc[1, 0:60]
record_b = dataset.iloc[2, 0:60]

location_c = random.randint(0, dataset.shape[0])
location_d = random.randint(0, dataset.shape[0])
record_c = dataset.iloc[location_c, 0:60]
record_d = dataset.iloc[location_d, 0:60]

mean_a, mean_b, mean_c, mean_d = [0.0] * 4
number_of_columns = len(record_a)
for index in range(number_of_columns):
    mean_a += record_a[index] / number_of_columns
    mean_b += record_b[index] / number_of_columns
    mean_c += record_c[index] / number_of_columns
    mean_d += record_d[index] / number_of_columns

variance_a, variance_b, variance_c, variance_d = [0.0] * 4
for index in range(number_of_columns):
    variance_a += (record_a[index] - mean_a) * (record_a[index] - mean_a) / number_of_columns
    variance_b += (record_b[index] - mean_b) * (record_b[index] - mean_b) / number_of_columns
    variance_c += (record_c[index] - mean_c) * (record_c[index] - mean_c) / number_of_columns
    variance_d += (record_d[index] - mean_d) * (record_d[index] - mean_d) / number_of_columns

correlation_ab, correlation_cd = [0.0] * 2
for index in range(number_of_columns):
    correlation_ab += (record_a[index] - mean_a) * (record_b[index] - mean_b) / (sqrt(variance_a * variance_b) * number_of_columns)
    correlation_cd += (record_c[index] - mean_c) * (record_d[index] - mean_d) / (sqrt(variance_c * variance_d) * number_of_columns)

print("Correlation between attributes 2 & 3:", correlation_ab)
print("Correlation between (random) attributes A-%s & B-%s: %s" % (location_c, location_d, correlation_cd))

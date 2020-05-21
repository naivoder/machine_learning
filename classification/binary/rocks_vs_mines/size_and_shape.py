"""
this file performs a simple list operation to return the shape of the given dataset
the dataset in this directory comes from the UC Irvice Data Repository

"""

from urllib.request import urlopen

host_url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data")
dataset = urlopen(host_url)

attributes = []; labels = []
for record in dataset:
    row = record.strip().decode().split(',')
    attributes.append(row)

number_of_rows = len(attributes)
number_of_columns = len(attributes[1])

print("Rows:", number_of_rows)
print("Columns:", number_of_columns)

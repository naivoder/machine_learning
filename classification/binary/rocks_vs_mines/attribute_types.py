"""
this file provides insight into the sample data attribute types by compiling a summary of numeric vs categorical inputs
here we see the dataset has 60 numeric attributes and a column of categorical labels

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

column_totals = []

for column in range(number_of_columns):
    # float, string, None
    type = [0] * 3
    for row in attributes:
        try:
            feature = float(row[column])
            if isinstance(feature, float):
                type[0] += 1
        except ValueError:
            if len(row[column]) > 0:
                type[1] += 1
            else:
                type[2] += 1
    column_totals.append(type)

index = 0
print("Column\tNumber\tString\tEmpty")
for types in column_totals:
    print("%s\t%s\t%s\t%s" % (index, types[0], types[1], types[2]))
    index += 1

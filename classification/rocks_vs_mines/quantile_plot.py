"""
this file aids the visualization of outliers in the data with a quantile plot, comparing the desired feature (default is random) quantile distribution against a normal Guassian distribution

"""

import pylab, random
import numpy as np
import scipy.stats as stats
from urllib.request import urlopen

host_url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data")
dataset = urlopen(host_url)

def quantile(dataset, feature=None):
    attributes = []
    for record in dataset:
        row = record.strip().decode().split(',')
        attributes.append(row)

    if feature is None:
        feature = random.randint(0, len(attributes[1]))

    # target is desired feature column
    target = feature; column_data = []
    for row in attributes:
        column_data.append(float(row[target]))

    stats.probplot(column_data, dist="norm", plot=pylab)
    pylab.show()

if __name__=="__main__":
    quantile(dataset)

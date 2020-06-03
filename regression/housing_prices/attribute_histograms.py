"""
this file uses matplotlib to plot a histogram for each attribute value

data for 'median_income', 'median_house_value', 'housing_median_age' have been capped
will need to consider dropping instances...
many attributes are tail-heavy, will try to normalize

"""
from pandas_summary import *
# magic jupyter function:
# %matplotlib inline
import matplotlib.pyplot as plt
housing_data.hist(bins=50, figsize=(20, 15))
plt.show()

"""
this file uses pandas to gain insight into the cali housing prices dataset

"""
import pandas as pd
from fetch_housing_data import *

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join('/home/naivoder/ml4py/datasets/housing', 'housing.csv')
    return pd.read_csv(csv_path)

housing_data = load_housing_data()
housing_data.info()

#notice that 'total_bedrooms' contains 207 instances of null data...
#all attributes are numerical except 'ocean_proximity', a categorical attributed
#need to find out what categories exist:
housing_data['ocean_proximity'].value_counts()

#info on numerical attributes
housing_data.describe()

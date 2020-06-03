"""
this file splits the data into test and training sets

"""
from pandas_summary import *
import numpy as np

#unnecessary custom function (sklearn has train_test_split())
def split_train_test(data, test_ratio=0.2):
    #shuffle data
    shuffled_indices = np.random.permutation(len(data))
    #standard 20%, use less for extra large datasets
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing_data)
print(len(train_set))

#above method will result in data leak if dataset is updated
#better practice to compute hash from identifier
#if hash < 20% maxhash --> instance belongs to test_set
from zlib import crc32

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio=0.2, id_column='index'):
    ids = data[id_column]
    #returns boolean array
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

#add index column to dataset
housing_with_id = housing_data.reset_index()
train_set, test_set = split_train_test_by_id(housing_with_id)

print(test_set.head())

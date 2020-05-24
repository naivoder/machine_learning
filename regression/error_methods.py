"""
this file contains very simple examples of regression algorithm performance measures
MSE (mean squared error), MAE (mean absolute error), and root MSE

"""
import random
from math import sqrt

batch = [random.randrange(-5, 5) for num in range(20)]
prediction = [random.randrange(-5, 5) for num in range(20)]

error = []
for i in range(len(batch)):
    error.append(batch[i] - prediction[i])
print("Errors ", error)

squaredError = []
absError = []
for val in error:
    squaredError.append(val*val)
    absError.append(abs(val))

print("Squared:", squaredError)
print("Absolute:", absError)
print("MSE =", sum(squaredError)/len(squaredError))
print("RMSE =", sqrt(sum(squaredError)/len(squaredError)))
print("MAE =", sum(absError)/len(absError))

batchDeviation = []
batchMean = sum(batch)/len(batch)
for val in batch:
    batchDeviation.append((val - batchMean)*(val - batchMean))

print("Batch Variance =", sum(batchDeviation)/len(batchDeviation))
print("Batch Standard Deviation =", sqrt(sum(batchDeviation)/len(batchDeviation)))

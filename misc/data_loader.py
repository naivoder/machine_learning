import os
import numpy as np
from PIL import Image

test_data = []

path = "/home/naivoder/ml4py/data"
for root, dirs, files in os.walk(path):
    for name in files:
        print(name)
        if name.endswith('PNG'):
            img = Image.open(os.path.join(root, name))
            data = np.asarray(img)
            print(data.shape)
            _name = name[:-3]
            _name += 'txt'
            answer_file = open(os.path.join(root, _name), 'r')
            answer = answer_file.read().rstrip()
            datapoint = (data, answer)
            test_data.append(datapoint)
print(test_data)
    # for name in files:
    #     if name.startswith('Basic') and name.endswith('PNG'):
    #         img = Image.open(name)
    #         data = np.asarray(img)
    #         print(data.shape)
    #     if name == 'ProblemAnswer.txt':
    #         answer_file = open(name, 'r')
    #         answer = answer_file.read()
    #     test_data.append((data, answer))
# print(test_data)

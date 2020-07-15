import os

file = open('trainval.txt', 'w')

path = '../images/'

for filename in os.listdir(path):
    id = filename.split('.')[0]
    file.write(id + '\n')

file.flush()
file.close()

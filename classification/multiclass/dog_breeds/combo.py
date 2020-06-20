import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import keras, math

from keras.applications import inception_v3, xception
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3, preprocess_input

from keras.layers import Dense, GlobalAveragePooling2D, Lambda, Dropout
from keras.models import Model

from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler

import cv2
from tqdm import tqdm

# Define input paths
train_folder = '../datasets/dog-breed-identification/train/'
test_folder = '../datasets/dog-breed-identification/test/'
df = pd.read_csv('../datasets/dog-breed-identification/labels.csv')

breed = set(df['breed'])

n = len(df)
breed = set(df['breed'])
n_class = len(breed)
class_to_num = dict(zip(breed, range(n_class)))
num_to_class = dict(zip(range(n_class), breed))

width = 299
batch_size = 64
epochs = 1000

X = np.zeros((n, width, width, 3), dtype=np.float16)
y = np.zeros((n, n_class), dtype=np.float16)
for i in tqdm(range(n)):
    X[i] = cv2.resize(cv2.imread(train_folder + '%s.jpg' % df['id'][i]), (width, width))
    y[i][class_to_num[df['breed'][i]]] = 1

def get_features(MODEL, data=X):
    cnn_model = MODEL(include_top=False, input_shape=(width, width, 3), weights='imagenet')
    inputs = keras.Input((width, width, 3))
    x = inputs
    x = Lambda(preprocess_input, name='preprocessing')(x)
    x = cnn_model(x)
    x = GlobalAveragePooling2D()(x)
    cnn_model = Model(inputs, x)
    features = cnn_model.predict(data, batch_size=batch_size, verbose=1)
    return features

inception_features = get_features(InceptionV3, X)
xception_features = get_features(Xception, X)

features = np.concatenate([inception_features, xception_features], axis=-1)
inputs = keras.Input(features.shape[1:])
x = inputs
x = Dropout(0.5)(x)
x = Dense(120, activation='softmax')(x)
model = Model(inputs, x)

# Learning rate schedule
def step_decay(epoch):
    init_rate = 0.0001
    drop = 0.9
    epochs_drop = 10
    lr = init_rate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lr

# Compile model with Adam optimizer
model.compile(Adam(lr=0.0), loss='categorical_crossentropy', metrics=['accuracy'])

# Set up model checkpoint, learning rate, early stopping callbacks
weight_path = 'weights/IncXc_619_.hdf5'
schedule = LearningRateScheduler(step_decay, verbose=1)
completion = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=100)
checkpoint = ModelCheckpoint(weight_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [completion, checkpoint, schedule]

# Train model and save history
history = model.fit(x=features,
                    y=y,
                    batch_size=batch_size,
                    validation_split=0.1,
                    epochs=epochs,
                    verbose=2,
                    callbacks=callbacks_list)

# Plot accuracy curves
plt.figure(0)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='upper left')
plt.savefig('IncXc_acc')

# Plot loss curves
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='upper left')
plt.savefig('IncXc_loss')

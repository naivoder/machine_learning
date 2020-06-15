"""
this file performs the classification task for the kaggle dog breed classification competition

"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

from keras.applications import inception_v3, densenet
from keras.applications.densenet import DenseNet121
from keras.applications.inception_v3 import InceptionV3

from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model

from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy

from sklearn.model_selection import train_test_split

# Define input paths
train_folder = '../datasets/dog-breed-identification/train/'
test_folder = '../datasets/dog-breed-identification/test/'
train_dogs = pd.read_csv('../datasets/dog-breed-identification/labels.csv')

# Only use top 20 breeds (if working in kaggle notebook)
# top_breeds = sorted(list(train_dogs['breed'].value_counts().head(20).index))
# train_dogs = train_dogs[train_dogs['breed'].isin(top_breeds)]

target_labels = train_dogs['breed']

# Convert to one hot encoded format
# one_hot = pd.get_dummies(target_labels, sparse=True)
# one_hot_labels = np.asarray(one_hot)

train_dogs['image_path'] = train_dogs.apply(lambda x: (train_folder + x["id"] + ".jpg"), axis=1)
train_dogs.head()

# InceptionV3 uses image sizes of 299 x 299
# train_data = np.array([img_to_array(load_img(img, target_size=(299, 299))) for img in train_dogs['image_path'].values.tolist()]).astype('float32')

# DenseNet121 uses image sizes of 224 x 224
train_data = np.array([img_to_array(load_img(img, target_size=(224, 224))) for img in train_dogs['image_path'].values.tolist()]).astype('float32')

# Split into training and validation sets
x_train, x_validation, y_train, y_validation = train_test_split(train_data, target_labels, test_size=0.2, stratify=np.array(target_labels), random_state=42)

# Convert the train and validation labels into one hot encoded format
y_train = pd.get_dummies(y_train.reset_index(drop=True)).values
y_validation = pd.get_dummies(y_validation.reset_index(drop=True)).values

# Define training generator
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=30,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip='true')
train_generator = train_datagen.flow(x_train, y_train, shuffle=True, batch_size=24, seed=3)

# Define validation generator
val_datagen = ImageDataGenerator(rescale = 1./255)
val_generator = train_datagen.flow(x_validation, y_validation, shuffle=True, batch_size=24, seed=3)

# Get the InceptionV3 model
# base_model = InceptionV3(weights = 'imagenet', include_top = False, input_shape=(299, 299, 3))

# Get the DenseNet121 model
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add pooling to collect soft features
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Add fully connected layer
x = Dense(512, activation='relu')(x)
predictions = Dense(120, activation='softmax')(x)

# Define training model
model = Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers:
    layer.trainable = True

# Compile model with Adam optimizer
model.compile(Adam(lr=.000001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train model and save history
history = model.fit_generator(train_generator,
                      steps_per_epoch=len(x_train)//24,
                      validation_data=val_generator,
                      validation_steps=len(x_validation)//24,
                      epochs=250,
                      verbose=2)

# Plot accuracy curves
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='upper left')
plt.show()

# Plot loss curves
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='upper left')
plt.show()

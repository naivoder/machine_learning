
import time
import urllib3
import subprocess
import numpy as np
import matplotlib.pylab as plt

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

from tensorflow.keras import layers

urllib3.disable_warnings()
tfds.disable_progress_bar()
tf.enable_eager_execution()

tf.logging.set_verbosity(tf.logging.ERROR)

(training, validation), info = tfds.load('cats_vs_dogs',
                                             split=['train[:80%]', 'train[80%:]'],
                                             with_info=True,
                                             as_supervised=True)

def format_image(image, label):
    image = tf.image.resize(image, (IMAGE, IMAGE))/255.0
    return image, label

num_examples = info.splits['train'].num_examples
print("Number of examples:", num_examples)

# mobilenet expects 224x224 images
BATCH = 32
IMAGE = 224

# # Image Input Pipeline
train_batches = training.cache().shuffle(num_examples//4).map(format_image).batch(BATCH).prefetch(1)
validate_batches = validation.cache().map(format_image).batch(BATCH).prefetch(1)

# # Define Feature Extractor
url = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'
feature_extractor = hub.KerasLayer(url, input_shape=(IMAGE, IMAGE, 3))

# # Freeze Base Model
feature_extractor.trainable = False

# # Add Output Layer
model = tf.keras.Sequential([
    feature_extractor,
    layers.Dense(2)
])

model.summary()

# # Train Model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

EPOCHS = 3

log = model.fit(train_batches, epochs=EPOCHS, validation_data=validate_batches)

# # Check Predictions
class_names = np.array(info.features['label'].names)
print(class_names)

image_batch, label_batch = next(iter(train_batches.take(1)))
image_batch = image_batch.numpy()
label_batch = label_batch.numpy()

predictions = model.predict(image_batch)
predictions = tf.squeeze(predictions).numpy()
prediction_labels = np.argmax(predictions, axis=-1)
prediction_names = class_names[prediction_labels]
print(prediction_names)

print("Actual   : ", label_batch)
print("Predicted: ", prediction_labels)

plt.figure(figsize=(10,9))
for n in range(30):
    plt.subplot(6,5,n+1)
    plt.imshow(image_batch[n])
    color = 'blue' if prediction_labels[n] == label_batch[n] else 'red'
    plt.title(prediction_names[n].title(), color=color)
    plt.axis('off')
_ = plt.suptitle('Model Predictions')

# # Save .h5 Model
t = time.time()
export_path = "./cat_dog_{}.h5".format(int(t))
print(export_path)

model.save(export_path)

# # Load .h5 Model
reload = tf.keras.models.load_model(
    export_path,
    # keras instructions for loading
    custom_objects = {'KerasLayer':hub.KerasLayer}
)

reload.summary()

# # Confirm Equality
results = model.predict(image_batch)
reload_results = reload.predict(image_batch)
(abs(results - reload_results)).max()

# # Continue Training
EPOCHS = 3
log = reload.fit(train_batches, epochs=EPOCHS, validation_data=validate_batches)

# # Export SavedModel
t = time.time()
model_export_path = "./cat_dog_{}".format(int(t))
print(export_path)

tf.saved_model.save(reload, model_export_path)

subprocess.run(['ls', '{model_export_path}'])

# # Load SavedModel
loaded = tf.compat.v2.saved_model.load(model_export_path)

loaded_result_batch = loaded(image_batch, training=False).numpy()

(abs(results - loaded_result_batch)).max()

# # Load SavedModel as Keras Model
saved_model = tf.keras.models.load_model(
    model_export_path,
    custom_objects={'KerasLayer': hub.KerasLayer}
)

model_results = reload.predict(image_batch)
saved_model_results = saved_model.predict(image_batch)

(abs(model_results - saved_model_results)).max()

# # Download Model
subprocess.run(['zip', '-r', 'model.zip', '{model_export_path}'])

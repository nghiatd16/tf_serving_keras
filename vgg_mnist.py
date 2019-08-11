# coding: utf-8
import tensorflow as tf
print("Tensorflow Version", tf.__version__)
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Model
import numpy as np
import os

# Config for gpu system
try:
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    keras.backend.set_session(tf.Session(config=config))
except:
    pass

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train, -1)/255.0
x_test = np.expand_dims(x_test, -1)/255.0

# Get data information
IMAGE_SHAPE = x_train.shape[1:]
TRAIN_SIZE = x_train.shape[0]
TEST_SIZE = x_test.shape[0]
EPOCHS = 10

# Define model architecture. A tiny VGG
def create_model():
    input_tensor = Input(IMAGE_SHAPE, name="input_tensor")
    net = Conv2D(filters=16, kernel_size=3, activation="relu", padding='same')(input_tensor)
    net = Conv2D(filters=16, kernel_size=3, activation="relu", padding='same')(net)
    net = MaxPool2D()(net)
    net = Dropout(0.5)(net)
    net = BatchNormalization()(net)
    net = Conv2D(filters=32, kernel_size=3, activation="relu", padding='same')(input_tensor)
    net = Conv2D(filters=32, kernel_size=3, activation="relu", padding='same')(net)
    net = MaxPool2D()(net)
    net = Dropout(0.5)(net)
    net = BatchNormalization()(net)
    
    net = Flatten()(net)
    net = Dense(units=128, activation="relu")(net)
    prediction = Dense(units=10, activation="softmax")(net)
    
    return Model(input_tensor, prediction)

vgg = create_model()
# Training
vgg.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
vgg.fit(x_train, y_train, epochs=EPOCHS, validation_data=(x_test, y_test))

# Define version of release model
version = 1
export_path = os.path.join('vggmodel', str(version))

# Export model to servable SavedModel format
tf.saved_model.simple_save(keras.backend.get_session(), export_path,
                            inputs={'input_images': model.input},
                            outputs={t.name:t for t in model.outputs})

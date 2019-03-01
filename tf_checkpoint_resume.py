#%% Import packages
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

#%% Initialize variables
checkpoint_path = './'

#%% Load and process datasets
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Reshape for CNNs
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# Cast to single precision
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#%% Define model
def mnist_mlp_model():
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(128, activation=tf.nn.relu),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    return model

def mnist_cnn_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=input_shape))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
    return model

#%% Load model


#%%
checkpoint_callback = ModelCheckpoint(filepath='weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss')
callbacks = [checkpoint_callback]

#%% Load checkpoint
if checkpoint_path is not None:
    # Load model:
    model = load_model(checkpoint_path)
    # Finding the epoch index from which we are resuming
    initial_epoch = get_init_epoch(checkpoint_path)
    # Calculating the correct value of count
    count = initial_epoch*batches_per_epoch

else:
    model = mnist_cnn_model()
    initial_epoch = 0

#%% Start/resume training
model.fit(x, y, callbacks=callbacks, initial_epoch=initial_epoch)
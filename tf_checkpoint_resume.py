#%% Import packages

import tensorflow as tf
import numpy as np
import os
import keras
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.models import Model
from download_cifar10 import load_data

#%% Training parameters
batch_size = 32  # orig paper trained all networks with batch_size=128
epochs = 200


#data_augmentation = False
num_classes = 10


#%% Initialize variables

dataset_path = '/Users/shshnkp/Projects/playground/datasets/'
checkpoint_path = '/Users/shshnkp/Projects/playground/checkpoints/'

#%% Load and prepare datasets
(x_train, y_train), (x_test, y_test) = load_data(dataset_path)
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Substract pixel mean
x_train_mean = np.mean(x_train, axis=0)
x_train -= x_train_mean
x_test -= x_train_mean

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#%% Define model


def cifar10_resnet50_model():
    input_tensor = Input(shape=input_shape)
    base_model = keras.applications.resnet50.ResNet50(include_top=False,
                                                      weights='imagenet',
                                                      input_tensor=input_tensor,
                                                      input_shape=input_shape,
                                                      classes=None)

    x = base_model.output
    x = Flatten()(x)
    predictions = Dense(10, activation='softmax')(x)
    mdl = Model(inputs=base_model.input, outputs=predictions)
    mdl.summary()
    return mdl

#%% Define Callbacks

# Model Checkpoint callback
model_name = 'cifar10_model.{epoch:03d}.h5'
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
filepath = os.path.join(checkpoint_path, model_name)
checkpoint_callback = ModelCheckpoint(filepath=filepath,
                                      monitor='val_loss')
callbacks = [checkpoint_callback]


#%% Train model

model = cifar10_resnet50_model()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=epochs, callbacks=callbacks)

#%%
model.evaluate(x_test, y_test)

#%% Start/resume training
model.fit(x, y, callbacks=callbacks, initial_epoch=initial_epoch)
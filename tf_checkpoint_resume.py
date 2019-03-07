#%% Import packages
import numpy as np
import os
import keras
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense, Flatten
from keras.models import Model, load_model
from download_cifar10 import load_data
from keras import backend as K

#%% Load and prepare datasets
def load_prepare_dataset(dataset_path):
    (x_train, y_train), (x_test, y_test) = load_data(dataset_path)

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

    return (x_train, y_train), (x_test, y_test)

#%% Define model
def cifar10_resnet50_model(input_shape):

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


#%%
def load_resume_model(checkpoint_path):
    list_of_checkpoint_files = os.listdir(checkpoint_path)
    checkpoint_epoch_number = max([int(file.split(".")[1]) for file in list_of_checkpoint_files])
    checkpoint_epoch_path = os.path.join(checkpoint_path,
                                         'cifar10_model.{:03d}.h5'.format(checkpoint_epoch_number))
    resume_model = load_model(checkpoint_epoch_path)
    return resume_model, checkpoint_epoch_number

#%%
def define_callbacks(checkpoint_path):
    # Model Checkpoint callback
    model_name = 'cifar10_model.{epoch:03d}.h5'
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)
    filepath = os.path.join(checkpoint_path, model_name)
    checkpoint_callback = ModelCheckpoint(filepath=filepath,
                                          save_weights_only=False,
                                          monitor='val_loss')
    callbacks = [checkpoint_callback]
    return callbacks


#%%
if __name__ == "__main__":

    # Training parameters
    batch_size = 32
    epochs = 50
    # data_augmentation = False
    num_classes = 10
    dataset_path = '/dltraining/datasets/'
    checkpoint_path = '/dltraining/checkpoints/'
    #dataset_path = '/Users/shshnkp/Projects/playground/datasets'
    #checkpoint_path = '/Users/shshnkp/Projects/playground/checkpoints'

    # Load dataset
    (x_train, y_train), (x_test, y_test) = load_prepare_dataset(dataset_path)
    input_shape = x_train.shape[1:]

    # %% Load model
    if os.path.isdir(checkpoint_path) and any(os.listdir(checkpoint_path)):
        model, epoch_number = load_resume_model(checkpoint_path)
    else:
        model = cifar10_resnet50_model(input_shape)
        epoch_number = 0

    # Define Callbacks
    callbacks = define_callbacks(checkpoint_path)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epochs, initial_epoch=epoch_number, callbacks=callbacks)

    #%%
    model.evaluate(x_test, y_test)

    #%% Start/resume training
    model.fit(x, y, callbacks=callbacks, initial_epoch=initial_epoch)


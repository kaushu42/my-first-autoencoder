import sys

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, UpSampling2D
from keras.callbacks import ModelCheckpoint

(x_train, _), (_, _) = keras.datasets.mnist.load_data()
# print(x_test.shape, x_test.shape)
x_train = x_train.reshape(-1, 28, 28, 1)

if sys.argv[1] == 'train':
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(UpSampling2D(size=(7, 7)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu'))
    print(model.summary())
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    # checkpoint
    filepath="./weights/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    # Fit the model
    model.fit(x_train, x_train, validation_split=0.33, epochs=100, batch_size=10, callbacks=callbacks_list, verbose=1)
    # print(model.evaluate(x_test))
elif sys.argv[1] =='test':
    model = load_model('./weights.h5')
    x = x_train[:10]
    y = model.predict(x).reshape(10, 28, 28)
    f, axarr = plt.subplots(10,2)

    for i in range(10):
        axarr[i, 0].imshow(x[i].reshape(28, 28), cmap='gray')
        axarr[i, 1].imshow(y[i].reshape(28, 28), cmap='gray')
    plt.show()

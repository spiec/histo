#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------
# Author:   sebastian.piec@
# Modified: 2019, March 12
# ----------------------------------------------------------------------

"""Simple convolutional neural nets to be trained from scratch on the given data.
"""

from keras import (layers, models,
                   optimizers)

from keras.models import Model, Sequential
from keras.layers import (Activation, Conv2D,
                          Dense, Dropout,
                          Flatten, Input,
                          MaxPooling2D)

# ----------------------------------------------------------------------
def cnn_V1(img_shape):
    """
    """
    inputs = Input(shape=img_shape)

    # encoder
    out = Conv2D(32, kernel_size=(3, 3), activation="relu")(inputs)
    out = MaxPooling2D(pool_size=(2, 2))(out)

    out = Conv2D(64, kernel_size=(3, 3), activation="relu")(out)
    out = MaxPooling2D(pool_size=(2, 2))(out)

    out = Conv2D(128, kernel_size=(3, 3), activation="relu")(out)
    out = MaxPooling2D(pool_size=(2, 2))(out)

    out = Conv2D(128, kernel_size=(3, 3), activation="relu", name="last_cnn")(out)
    out = MaxPooling2D(pool_size=(2, 2))(out)

    # decoder
    out = Flatten()(out)

    out = Dropout(0.5)(out)
    out = Dense(512, activation="relu")(out)
    out = Dense(2, activation="softmax")(out)

    model = Model(inputs, outputs=out)

    # TODO tune optimizer's params
    model.compile(loss="binary_crossentropy",
                  optimizer=optimizers.Adam(lr=1e-4),
                  metrics=["acc"])

    return model

# ----------------------------------------------------------------------
def cnn_V2(image_shape):
    kernel_size = (3, 3)
    pool_size= (2, 2)
    first_filters = 32
    second_filters = 64
    third_filters = 128

    dropout_conv = 0.3
    dropout_dense = 0.3

    model = Sequential()
    model.add(Conv2D(first_filters, kernel_size, activation = 'relu', input_shape=image_shape))
    model.add(Conv2D(first_filters, kernel_size, activation = 'relu'))
    model.add(Conv2D(first_filters, kernel_size, activation = 'relu'))
    model.add(MaxPooling2D(pool_size = pool_size)) 
    model.add(Dropout(dropout_conv))

    model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
    model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
    model.add(Conv2D(second_filters, kernel_size, activation ='relu'))
    model.add(MaxPooling2D(pool_size = pool_size))
    model.add(Dropout(dropout_conv))

    model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
    model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
    model.add(Conv2D(third_filters, kernel_size, activation ='relu'))
    model.add(MaxPooling2D(pool_size = pool_size))
    model.add(Dropout(dropout_conv))

    model.add(Flatten())
    model.add(Dense(256, activation = "relu"))
    model.add(Dropout(dropout_dense))
    model.add(Dense(2, activation = "softmax"))

    model.compile(loss="binary_crossentropy",
                  optimizer=optimizers.Adam(lr=1e-4),
                  metrics=["acc"])

    return model
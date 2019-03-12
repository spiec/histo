#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------
# Author:   sebastian.piec@
# Modified: 2019, March 12
# ----------------------------------------------------------------------

"""Use convolutional nets (pretrained on the ImageNet data)
as a static feature extractors, with dense NN trained on top
(transfer learning).
"""

from keras import applications, optimizers
from keras.layers import (Activation, BatchNormalization, Concatenate, Dense, Dropout,
                          Flatten, GlobalAveragePooling2D, GlobalMaxPooling2D)
from keras.models import Input, Model, load_model

# ----------------------------------------------------------------------
class ConvLearner(object):

    def __init__(self, base_model, image_shape, optimizer=None, metrics=["acc"]):
        input_tensor = Input(shape=image_shape)
        self.conv_base = self.make_base(base_model, image_shape)

        self.image_shape = image_shape
        self.optimizer = optimizer
        self.metrics = metrics
        
        #bn = BatchNormalization()(input_tensor)
        #x = self.conv_base(bn)
        x = self.conv_base(input_tensor)
        
        self._head = self.make_head(x)

        self.model = Model(input_tensor, self._head)
        self.recompile()
        
        # to possibly retrain a few last layers
        self.conv_trainable = [-1]      #, -2]

    def freeze(self, module):
        if module == "conv_top":
            for layer in self.conv_base.layers:
                if layer.name in ["conv2d_94", "batch_normalization_95"]:
                    layer.trainable = False
                    break
        elif module == "head":
            for layer in self.model.layers:
                if layer.name.startswith("dense_head_"):
                    layer.trainable = False

    def unfreeze(self, module):
        if module == "conv_top":
            for layer in self.conv_base.layers:
                if layer.name in ["conv2d_94", "batch_normalization_95"]:
                    print("unfreeze", layer.name)
                    layer.trainable = True
                    break
        elif module == "head":
            print("unfreeze head")
            #self._head.trainable = True
            for layer in self.model.layers:
                if layer.name.startswith("dense_head_"):
                    layer.trainable = True

    def recompile(self):
        self.model.compile(loss="binary_crossentropy",
                           optimizer=self.optimizer,
                           metrics=self.metrics) 

    def make_base(self, base_model, image_shape):
        conv_base = _base_models[base_model](image_shape)
        print("Base model # of layers:", len(conv_base.layers))
        return conv_base

    def make_head(self, x):
        """Typically dense neural network on top of the convolutional layers.
        """
        out1 = GlobalMaxPooling2D()(x)
        out2 = GlobalAveragePooling2D()(x)
        out3 = Flatten()(x)
        out = Concatenate(axis=-1)([out1, out2, out3])
        
        out = Dropout(0.5)(out)
        output = Dense(2, activation="softmax", name="dense_head_2")(out)

        return output

    def load(self, filename):
        print("Loading model {}".format(filename), type(filename))
        self.model = load_model(filename)

# ----------------------------------------------------------------------
def _nasnetmobile(image_shape):
    return applications.NASNetMobile(include_top=False,
                                     input_shape=image_shape)

def _inceptionv3(image_shape):
    return applications.InceptionV3(include_top=False,
                                    weights="imagenet",
                                    input_shape=image_shape)

def _resnet50(image_shape):
    return applications.ResNet50(include_top=False,
                                 weights="imagenet",
                                 input_shape=image_shape)

def _vgg16(image_shape):
    return applications.VGG16(include_top=False,
                              weights="imagenet",
                              input_shape=image_shape)

def _vgg19(image_shape):
    return applications.VGG19(include_top=False,
                              weights="imagenet",
                              input_shape=image_shape)

def _xception(image_shape):
    return applications.Xception(include_top=False,
                                 weights="imagenet",
                                 input_shape=image_shape)

def _mobilenet(image_shape):
    return applications.MobileNet(include_top=False,
                                  weights="imagenet",
                                  input_shape=image_shape)

_base_models = {
        "nasnet_mobile": _nasnetmobile,
        "inceptionv3": _inceptionv3,
        "resnet50": _resnet50,
        "xception": _xception,
        "mobilenet": _mobilenet,
        "vgg16": _vgg16,
        "vgg19": _vgg19,
    }

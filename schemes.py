#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------
# Author:   sebastian.piec@
# Modified: 2019, March 12
# ----------------------------------------------------------------------

"""Different training schemes.
"""

from time import time

from keras import optimizers
from keras.callbacks import (CSVLogger,
                             EarlyStopping,
                             ModelCheckpoint,
                             ReduceLROnPlateau,
                             RemoteMonitor,
                             TensorBoard)

import config
from modelzoo import base, pretrained
#from utils import CyclicLR

# ----------------------------------------------------------------------
def train_model(train_generator, eval_generator, output_file):
    callbacks = [ModelCheckpoint(filepath=output_file, verbose=1,
                                 save_best_only=True),
                 EarlyStopping(monitor="val_acc", min_delta=0.0001, patience=10),
                 ReduceLROnPlateau(monitor="val_loss", factor=0.2, verbose=1,
                                   patience=5, min_lr=0.0001),
                 #TensorBoard(log_dir="logs/{}".format(time())),
                 CSVLogger("./logs/last_run.log"),
                 #CyclicLR(),
                ]

    learner = pretrained.ConvLearner("nasnet_mobile", config.image_shape, optimizers.Adam(lr=0.0001))
    model = learner.model
    print(model.summary())

    model, history = _varbatch_training(model, train_generator, eval_generator, callbacks)
    #model.save(model_file)

    return model, history

# ----------------------------------------------------------------------
def _simple_training(model, train_generator, eval_generator, callbacks):
    history = model.fit_generator(train_generator,
                                  epochs=config.epochs,
                                  validation_data=eval_generator,           # or eval_split
                                  validation_steps=10,
                                  callbacks=callbacks,
                                  verbose=1)
    return model, history

# ----------------------------------------------------------------------
def _varbatch_training(model, train_generator, eval_generator, callbacks):
    # variable batch size training
    train_generator.batch_size = 32
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=len(train_generator),
                                  epochs=2,
                                  validation_data=eval_generator,
                                  validation_steps=10,
                                  callbacks=callbacks,
                                  verbose=1)

    train_generator.batch_size = 64
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=len(train_generator),
                                  epochs=6,
                                  validation_data=eval_generator,
                                  validation_steps=10,
                                  callbacks=callbacks,
                                  verbose=1)

    print("Reloading best weights {}".format(config.model))
    model.load_weights(config.model)

    return model, history

# ----------------------------------------------------------------------
def cfu_training(train_generator, train_labels, eval_generator, output_file):
    """Cyclic freeze/unfreeze head and top convolutional base.

    [conv_base] ==> [conv_top] ==> [dense_head]

    """
    callbacks = [ModelCheckpoint(filepath=output_file, verbose=1,
                                 save_best_only=True),
                 EarlyStopping(monitor="val_acc", min_delta=0.0001, patience=5),
                 ReduceLROnPlateau(monitor="val_loss", factor=0.2, verbose=1,
                                   patience=5, min_lr=0.0001),
                 RemoteMonitor(root='http://localhost:9000'),
                 CSVLogger("./logs/last_run.log"),
                ]


    n_steps = len(train_labels) / config.batch_size
    fu_steps = 1                # number of freeze-unfreeze cycles

    learner = pretrained.ConvLearner("inceptionv3", "adam", config.image_shape)

    for idx in range(fu_steps):
        # (a) train classifier on-top
        print("[Cycle {}]".format(idx))
        learner.freeze("conv_top")
        learner.unfreeze("head")

        learner.recompile()
        print(learner.model.summary())

        history = learner.model.fit_generator(train_generator,
                                              steps_per_epoch=n_steps,
                                              epochs=1,
                                              validation_data=eval_generator,           # or eval_split
                                              validation_steps=10,
                                              callbacks=callbacks,
                                              verbose=1)

        # (2) fine-tune middle convolutional layers
        learner.freeze("head")
        learner.unfreeze("conv_top")
        learner.optimizer = optimizers.RMSprop(lr=1e-5)

        learner.recompile()
        print(learner.model.summary())
        #print("trainable weights", len(model.trainable_weights))

        # NOTE: last conv layers of the conv_base require more gentle optimization typically
        #model.optimizer = 

        history = learner.model.fit_generator(train_generator,
                                  steps_per_epoch=n_steps,
                                  epochs=1,
                                  validation_data=eval_generator,           # or eval_split
                                  validation_steps=10,
                                  callbacks=callbacks,
                                  verbose=1)

    print("[Retrain head...]")

    # retrain head and top convolutional layers
    learner.unfreeze("head")
    learner.recompile()

    history = learner.model.fit_generator(train_generator,
                                          steps_per_epoch=n_steps,
                                          epochs=config.epochs,
                                          validation_data=eval_generator,
                                          validation_steps=10,
                                          callbacks=callbacks,
                                          verbose=1)


    return learner.model

# ----------------------------------------------------------------------
if __name__ == "__main__":
    pass
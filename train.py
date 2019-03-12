#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------
# Author:   sebastian.piec@
# Modified: 2019, March 12
# ----------------------------------------------------------------------

"""Binary image classification for Histopathologic Cancer Detection
(https://www.kaggle.com/c/histopathologic-cancer-detection)

Usage:
    python train.py --epochs=10
"""

import config
from data import data_generators
from histoutils import eval_model, training_summary
from modelzoo import pretrained
from schemes import train_model

# ----------------------------------------------------------------------
def train():
    train_generator, eval_generator = data_generators(config.train_dir)
    print("Training batches: {}, eval batches: {}".format(len(train_generator), len(eval_generator)))

    model, history = train_model(train_generator, eval_generator, output_file=config.model)
    training_summary(history, model, train_generator, eval_generator)

# ----------------------------------------------------------------------
def evaluate(args):
    train_generator, eval_generator = data_generators(config.train_dir)
    
    learner = pretrained.ConvLearner("nasnet_mobile", config.image_shape)
    learner.model.load_weights(config.model)

    eval_model(learner.model, train_generator, eval_generator)

# ----------------------------------------------------------------------
if __name__ == "__main__":
    train()
    #evaluate()

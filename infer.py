#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------
# Author:   sebastian.piec@
# Modified: 2019, March 12
# ----------------------------------------------------------------------

"""
Usage:
    python infer.py --model=./output/nasnet_v0.h5
"""

import config
from data import test_data_generator
from histoutils import inference
from modelzoo import pretrained

# ----------------------------------------------------------------------
def infer():
    data_generator = test_data_generator(config.test_dir)

    learner = pretrained.ConvLearner("nasnet_mobile", config.image_shape, "adam")
    learner.model.load_weights(config.model)

    inference(learner.model, data_generator)

# ----------------------------------------------------------------------
if __name__ == "__main__":
    infer()
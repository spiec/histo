#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------
# Author:   sebastian.piec@
# Modified: 2019, March 12
# ----------------------------------------------------------------------

seed = 999

batch_size = 32
epochs = 10
eval_size = 0.025

train_dir = "./input/train"
test_dir = "./input/test"
model = "./output/nasnet_v0.h5"         # trained model
submission = "./output/submission.csv"

image_shape = (96, 96, 3)

# ----------------------------------------------------------------------
def read_args():
    global seed, epochs, model

    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-s", "--seed", type=int, default=seed,
                        help="random seed")
    parser.add_argument("-e", "--epochs", type=int, default=epochs,
                        help="# of training epochs")
    parser.add_argument("-m", "--model", type=str, default=model,
                        help="model file")
    args = parser.parse_args()

    seed = args.seed
    epochs = args.epochs
    model = args.model
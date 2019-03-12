#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------
# Author:   sebastian.piec
# Modified: 2019, March 12
# ----------------------------------------------------------------------

import os

import cv2
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from skimage import io
from skimage.util import random_noise

import keras
#from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

import config
from imgaugment import augment_images

# ----------------------------------------------------------------------
class HistoDataGenerator(keras.utils.Sequence):
    """
    Credits:
    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    """

    def __init__(self, input_dir, file_names, n_classes, batch_size, image_shape,
                 shuffle=False, labels_file=None, augmentor=None):
        super(HistoDataGenerator, self).__init__()

        self.input_dir = input_dir
        self.file_names = file_names
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.shuffle = shuffle

        self.labels_file = labels_file
        if labels_file:
            self.labels_map = self._extract_labels(file_names, labels_file)

        self.augmentor = augmentor
        self.indexes = self._reset_indexes()

    def __getitem__(self, index):
        """Return single data batch in form of (X, y) tuple.
        TODO (optionally) return stratified sample
        """
        start = index * self.batch_size
        batch_size = min(self.batch_size, len(self.indexes) - start)

        indexes = self.indexes[start: start + batch_size]
        sublist = [self.file_names[k] for k in indexes]
        
        if self.labels_file:
            return self._generate_train_batch(sublist, batch_size)
        
        return (self._generate_test_batch(sublist, batch_size),
                [os.path.splitext(filename)[0] for filename in sublist])

    def __len__(self):
        """Returns number of data batches.
        """
        return len(self.indexes) // self.batch_size + (1 if len(self.indexes) % self.batch_size > 0 else 0)

    def _generate_train_batch(self, file_names, batch_size):
        X_batch = np.zeros((batch_size, *self.image_shape))
        y_batch = np.zeros(batch_size)

        for idx, filename in enumerate(file_names):
            X_batch[idx, :] = self._load_image(os.path.join(self.input_dir, filename))
            y_batch[idx] = self.labels_map[filename]

        if self.augmentor:
            X_batch = self.augmentor(X_batch)
        
        X_batch = self._preprocess_batch(X_batch)
        y_batch = keras.utils.to_categorical(y_batch, num_classes=self.n_classes)       # 1-hot encode

        return X_batch, y_batch

    def _generate_test_batch(self, file_names, batch_size):
        X_batch = np.zeros((batch_size, *self.image_shape))

        for idx, filename in enumerate(file_names):
            X_batch[idx, :] = self._load_image(os.path.join(self.input_dir, filename))
        X_batch = self._preprocess_batch(X_batch)

        return X_batch

    def _load_image(self, filename):
        #return cv2.imread(filename)
        return image.load_img(filename, target_size=self.image_shape[:2])

    def _extract_labels(self, file_names, labels_file):
        labels_map = {}
        df = pd.read_csv(labels_file, index_col="id")

        for filename in tqdm(file_names):
            file_id = filename.split(".")[-2]
            labels_map[filename] = df.loc[file_id]

        return labels_map

    def on_epoch_end(self):
        self.indexes = self._reset_indexes()

    def _reset_indexes(self):
        indexes = np.arange(len(self.file_names))

        if self.shuffle:
            #np.random.shuffle(indexes)
            indexes = shuffle(indexes, random_state=config.seed)
        return indexes

    def _preprocess_batch(self, image_batch):
        for i in range(image_batch.shape[0]):
            image_batch[i] = image.img_to_array(image_batch[i])
            image_batch[i] /= 255.

        return image_batch

# ----------------------------------------------------------------------
def data_generators(input_dir):
    """Returns training and evaluation data generators.
    """
    image_list = [f for f in sorted(os.listdir(input_dir)) if
                  os.path.isfile(os.path.join(input_dir, f)) and f.endswith(".tif")]
    #image_list = image_list[:1000]

    split_point = int(len(image_list) * (1. - config.eval_size))

    train_files = image_list[:split_point]
    eval_files = image_list[split_point:]

    labels_file = os.path.join(input_dir, "../train_labels.csv")

    train_generator = HistoDataGenerator(input_dir, train_files, 2,
                                         config.batch_size, config.image_shape,
                                         shuffle=True, labels_file=labels_file, augmentor=augment_images)

    eval_generator = HistoDataGenerator(input_dir, eval_files, 2,
                                        config.batch_size, config.image_shape,
                                        shuffle=True, labels_file=labels_file)

    return train_generator, eval_generator

# ----------------------------------------------------------------------
def test_data_generator(input_dir):
    image_list = [f for f in sorted(os.listdir(input_dir)) if
                  os.path.isfile(os.path.join(input_dir, f)) and f.endswith(".tif")]

    return HistoDataGenerator(input_dir, image_list, 2,
                              config.batch_size, config.image_shape,
                              shuffle=False)

# ----------------------------------------------------------------------
def _show_items(data_generator, figure_title="Data", augmentor=None):
    X_batch, y_batch = data_generator[0]
    n = min(len(X_batch), 24)

    rows = 4
    cols = n // rows + (1 if n % rows else 0)
    print(rows, cols, n)
    fig, axes = plt.subplots(rows, cols, figsize=(20, 18))
    axes = np.ravel(axes)

    for i in range(n):
        img, y = X_batch[i], y_batch[i]
        if augmentor:
            img = augmentor(img)
        axes[i].imshow(img)
        axes[i].set_title("Label {}".format(y[0]))

    fig.suptitle(figure_title)
    plt.show()

# ----------------------------------------------------------------------
def _show_augmentation(data_generator, augmentor):
    X_batch, y_batch = data_generator[0]

    n_rows = 4
    n_transforms = 1 + len(augmentor)

    fig, axes = plt.subplots(n_rows, n_transforms, figsize=(14, 8))
    #axes = np.ravel(axes)

    for r in range(n_rows):
        img, y = X_batch[r], y_batch[r]
        axes[r, 0].imshow(img)
        img = augmentor(img)
        axes[r, 1].imshow(img)

    fig.suptitle("Image Augmentation Tests")
    plt.show()    

# ----------------------------------------------------------------------
def generator_tests():
    train_generator, eval_generator = data_generators(config.train_dir)
    print("Training batches: {}, eval batches: {}".format(len(train_generator), len(eval_generator)))

    _show_items(train_generator, "Training Data")
    _show_items(eval_generator, "Evaluation Data")

    #from imgaugment import TrainAugmentor
    #_show_augmentation(train_generator, TrainAugmentor())

# ----------------------------------------------------------------------
if __name__ == "__main__":
    generator_tests()
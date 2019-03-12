#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------
# Author:   sebastian.piec@
# Modified: 2019, March 12
# ----------------------------------------------------------------------

"""Training data augmentation.
"""

from functools import partial
import numpy as np

from skimage.transform import rotate

import imgaug as ia
from imgaug import augmenters as iaa

# ----------------------------------------------------------------------
def augment_images(images_batch):
    #return _advanced.augment_images(images_batch)
    return _simple.augment_images(images_batch)

sometimes = lambda aug: iaa.Sometimes(0.5, aug)

_advanced = iaa.Sequential(
        [
            iaa.Fliplr(0.5),
            iaa.Flipud(0.2), 
            sometimes(iaa.Affine(
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                rotate=(-10, 10),
                shear=(-5, 5),
                order=[0, 1],
                cval=(0, 255),
                mode=ia.ALL
            )),
            iaa.SomeOf((0, 5),
                [
                    #sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 1.0)),
                        iaa.AverageBlur(k=(3, 5)),
                        #iaa.MedianBlur(k=(3, 5)),
                    ]),
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.9, 1.1)),
                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),

                    iaa.SimplexNoiseAlpha(iaa.OneOf([
                        iaa.EdgeDetect(alpha=(0.5, 1.0)),
                        iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                    ])),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01*255), per_channel=0.5),
                    iaa.OneOf([
                        iaa.Dropout((0.01, 0.05), per_channel=0.5),
                        iaa.CoarseDropout((0.01, 0.03), size_percent=(0.01, 0.02), per_channel=0.2),
                    ]),
                    iaa.Invert(0.01, per_channel=True),
                    iaa.Add((-2, 2), per_channel=0.5),
                    #iaa.AddToHueAndSaturation((-1, 1)),
                    iaa.OneOf([
                        iaa.Multiply((0.9, 1.1), per_channel=0.5),
                        iaa.FrequencyNoiseAlpha(
                            exponent=(-1, 0),
                            first=iaa.Multiply((0.9, 1.1), per_channel=True),
                            second=iaa.ContrastNormalization((0.9, 1.1))
                        )
                    ]),
                    sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
                    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                    sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                ],
                random_order=True
            )
        ],
        random_order=True
    )

# ----------------------------------------------------------------------
_simple = iaa.Sequential([
            #iaa.Crop(px=(0, 16)), 
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),

            #iaa.Sometimes(0.2, 
            #    iaa.GaussianBlur(sigma=(0, 0.5)),
            #),
            iaa.Sharpen(alpha=(0, 0.1), lightness=(0.9, 1.1)),

            iaa.OneOf([
                #iaa.GaussianBlur((0, 0.2)),
                #iaa.AverageBlur(k=(2, 7)),
                #iaa.MedianBlur(k=(3, 11)),
            ]),
            #iaa.Multiply((0.8, 1.2), per_channel=0.5),
        ])

# ----------------------------------------------------------------------
def horizontal_flip(p=0.5):
    def f(img, p):
        if np.random.rand() < p:
            return img[:, ::-1]
        return img

    return partial(f, p=p)

# ----------------------------------------------------------------------
def vertical_flip(p=0.5):
    def f(img, p):
        if np.random.rand() < p:
            return img[::-1, :]
        return img

    return partial(f, p=p)

# ----------------------------------------------------------------------
def random_rotate(angle=10.0):
    angle = np.random.rand() * np.abs(angle)
    return partial(rotate, angle=angle)
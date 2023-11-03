#!/usr/bin/env python3
"""Loads & Preprocesses the
CIFAR10 Dataset from Keras"""

import numpy as np
from keras.datasets import cifar10


def load_data():
    """Load & Preprocess CIFAR10 dataset"""
    (X_train, _), (X_test, _) = cifar10.load_data()
    X = np.vstack((X_train, X_test))
    X = X.astype('float 32')
    
    # Normalize: [-1, 1]
    X = (X - 127.5) / 127.5
    
    # Normalize to [0, 1]
    # X = X/255.0

    return X

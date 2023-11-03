#!/usr/bin/env python3
"""Loads & Preprocesses the
MNIST Dataset from Keras"""

import numpy as np
from keras.datasets import mnist


def load_data_set():
    """Load & Preprocess MNIST dataset"""
    (X_train, _), (X_test, _) = mnist.load_data()
    X = np.vstack((X_train, X_test))
    X = X.astype('float 32')
    
    # Normalize: [-1, 1]
    X = (X - 127.5) / 127.5
    
    # Normalize to [0, 1]
    # X = X/255.0

    return X

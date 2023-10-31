# preprocess.py
import numpy as np
import glob
from keras.preprocessing.image import load_img, img_to_array

def load_cartoon_set():
    X = []
    for image_path in glob.glob('path_to_cartoon_set/*.png'):
        image = img_to_array(load_img(image_path))
        X.append(image)
    X = np.array(X).astype('float32')
    # Normalize to [-1, 1]
    X = (X - 127.5) / 127.5

    # Normalize to [0, 1]
    # X = X/255.0

    return X

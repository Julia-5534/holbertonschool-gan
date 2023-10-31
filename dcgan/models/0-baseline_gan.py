#!/usr/bin/env python3
"""Baseline Generative Adversarial Model"""

import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Input, LeakyReLU, Reshape
from keras.models import Model, Sequential
from keras.optimizers import Adam
from tqdm import tqdm_notebook
from data.preprocess import load_data_set
import wandb

# Load the preprocessed data
X = load_data_set()

# Initialize a new wandb run
wandb.init(project="base-gan-mnist")

def discriminator():
  input = Input(shape=(28, 28, 1))

  x = Flatten()(input)
  x = Dropout(0.4)(x)
  x = Dense(1024, activation=LeakyReLU(alpha=0.2))(x)
  x = Dropout(0.4)(x)
  x = Dense(512, activation=LeakyReLU(alpha=0.2))(x)
  x = Dropout(0.4)(x)
  x = Dense(512, activation=LeakyReLU(alpha=0.2))(x)

  output = Dense(1, activation='sigmoid')(x)

  model = Model(input, output)
  model.compile(optimizer=Adam(lr=0.0002, beta_1=0.5), loss='binary_crossentropy')

  return model


def generator(n):
  input = Input(shape=(n))

  x = Dense(256, activation=LeakyReLU(alpha=0.2))(input)
  x = Dense(512, activation=LeakyReLU(alpha=0.2))(x)
  x = Dense(1024, activation=LeakyReLU(alpha=0.2))(x)
  x = Dense(784, activation='tanh')(x)

  output = Reshape((28, 28, 1))(x)

  return Model(input, output)


def gan(dis, gen):
  dis.trainable = False

  model = Sequential()

  model.add(gen)
  model.add(dis)

  model.compile(optimizer=Adam(lr=0.0002, beta_1=0.5), loss='binary_crossentropy')

  return model

discrim = discriminator()

geney = generator(100)

gan_model = gan(discrim, geney)


epochs = 80
batch_size = 256
half_batch = batch_size // 2
n = 100

for i in range(epochs):
    print("EPOCH", i)
    for j in tqdm_notebook(range(len(X) // batch_size)):
        x_real, y_real = X[np.random.randint(0, len(X), half_batch)].reshape(half_batch, 28, 28, 1), np.ones(half_batch).reshape(half_batch, 1)
        x_fake, y_fake = geney.predict(np.random.randn(half_batch, n)), np.zeros(half_batch).reshape(half_batch, 1)
        x_final, y_final = np.vstack((x_real, x_fake)), np.vstack((y_real, y_fake))
        dis_loss = discrim.train_on_batch(x_final, y_final)
        gen_loss = gan_model.train_on_batch(np.random.randn(batch_size, n), np.ones(batch_size).reshape(batch_size, 1))

        # Log losses to wandb
        wandb.log({"Discriminator Loss": dis_loss,
                   "Generator Loss": gen_loss})

    print("Discriminator Loss:", dis_loss)
    print("Generator Loss:", gen_loss)

    if i % 10 == 0:
        fig, axes = plt.subplots(5, 5, figsize=(12, 12))
        images = []
        for ii in range(5):
            for jj in range(5):
                img = geney.predict(np.random.randn(1 * n).reshape(1, n)).reshape(28, 28)
                axes[ii, jj].imshow(img, cmap='gray')
                images.append(wandb.Image(img))  # Log image to W&B
        plt.show()
        plt.close()
        wandb.log({"Generated Images": images})  # Log all images to W&B

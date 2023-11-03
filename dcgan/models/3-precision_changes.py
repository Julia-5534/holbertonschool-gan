#!/usr/bin/env python3
"""Experiment 3: Deep Convolutional
Generative Adversarial Model (DCGAN)"""

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Input, LeakyReLU, Reshape, ReLU
from keras.layers import Activation, Dense, Flatten, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from tqdm import tqdm_notebook
from keras.layers import Conv2D, Conv2DTranspose
import os
import wandb

# Load the dataset
(X_train, _), (X_test, _) = mnist.load_data()
X = np.vstack((X_train, X_test))
X = X.astype('float16')

# Normalize: [-1, 1]
X = (X - 127.5) / 127.5

# Initialize a new wandb run
wandb.init(project="DCGAN", name="exp3")

def discriminator():
    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # model.add(Dropout(0.4))
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(lr=0.0001, beta_1=0.5), loss='binary_crossentropy')

    return model

def generator(n):
    model = Sequential()
    model.add(Dense(7*7*128, input_dim=n))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Reshape((7, 7, 128)))
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Conv2DTranspose(1, (4, 4), strides=(2, 2), padding='same'))
    model.add(Activation('tanh'))

    return model

def gan(dis, gen):
    dis.trainable = False

    model = Sequential()
    model.add(gen)
    model.add(dis)

    model.compile(optimizer=Adam(lr=0.0002, beta_1=0.5), loss='binary_crossentropy')

    return model

# Create the logs/baseline directory if it doesn't exist
if not os.path.exists('logs/exp3'):
    os.makedirs('logs/exp3')

discrim = discriminator()
geney = generator(100)
gan_model = gan(discrim, geney)

epochs = 50
batch_size = 128
half_batch = batch_size // 2
n = 100

for i in range(epochs):
    print("EPOCH", i)
    for j in tqdm_notebook(range(len(X) // batch_size)):
        # Generate random noise
        noise = np.random.normal(0, 1, [half_batch, n])

        # Generate fake images
        x_fake = geney.predict(noise)

        # Use soft labels for training the discriminator
        y_real_soft = np.random.uniform(0.9, 1.0, size=(half_batch,))
        y_fake_soft = np.random.uniform(0.0, 0.1, size=(half_batch))

        # Train discriminator on real and fake data separately
        x_real = X[np.random.randint(0, len(X), half_batch)].reshape(half_batch, 28, 28, 1)
        d_loss_real = discrim.train_on_batch(x_real + np.random.normal(loc=0.0,scale=0.05,size=x_real.shape), y_real_soft)
        d_loss_fake = discrim.train_on_batch(x_fake + np.random.normal(loc=0.0,scale=0.05,size=x_fake.shape), y_fake_soft)

        # Calculate the total discriminator loss
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Generate new noise for the generator
        noise = np.random.normal(0, 1, [batch_size, n])
        valid_y = np.array([1] * batch_size)

        # Train the generator within the GAN model
        g_loss = gan_model.train_on_batch(noise + np.random.normal(loc=0.0,scale=0.05,size=noise.shape), valid_y)

    print("Discriminator Loss:", d_loss)
    print("Generator Loss:", g_loss)
    
    # Log the losses to wandb
    wandb.log({"Discriminator Loss": d_loss, "Generator Loss": g_loss})

    fig, axes = plt.subplots(5, 5)
    images = []
    for ii in range(5):
        for jj in range(5):
            img = geney.predict(np.random.randn(1, n)).reshape(28, 28)
            axes[ii, jj].imshow(img, cmap='gray')
            images.append(wandb.Image(img))
    save_path = os.path.join('logs', 'exp3', f'image_at_epoch_{i:04d}.png')
    plt.show()
    plt.close()
    wandb.log({"EXP3 Generated Images": images})

wandb.finish()

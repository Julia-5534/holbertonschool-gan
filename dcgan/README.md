<p align="center">
  <Deep Convolutional Generative Adversarial Networks>
</p>


<p align="center">
<br>A Good Ol' DCGAN&trade; By</br>
Julia Bullard
</p>


## :book: Table of Contents :book:
* [Key Concepts](#key-concepts)
* [Environment](#environment)
* [Requirements](#requirements)
* [Baseline Code Structure](#baseline-code-structure)
* [The Experiments](#the-experiments)
* [Authors](#authors)
* [License](#license)

## :key: :brain: Key Concepts :brain: :key:
* Generative Adversarial Networks (GANs):
     * GANs are a class of machine learning models that consist of two neural networks: a generator and a discriminator. The generator generates synthetic data, while the discriminator tries to distinguish between real and fake data. The two networks are trained simultaneously, with the goal of the generator producing data that is indistinguishable from real data.

* Deep Convolutional Generative Adversarial Network (DCGAN):
     * DCGAN is an extension of GANs that uses Convolutional Neural Networks (CNNs) in both the generator and discriminator. CNNs are particularly well-suited for image generation tasks, as they can capture spatial dependencies in the data.

* Discriminator:
     * The discriminator network is responsible for distinguishing between real and fake images. It takes an input image and outputs a probability indicating whether the image is real or fake. In the provided code, the `discriminator()` function defines the architecture of the discriminator network.

* Generator:
     * The generator network takes random noise as input and generates synthetic images. The goal of the generator is to produce images that are realistic enough to fool the discriminator. In the code, the `generator(n)` function defines the architecture of the generator network.

* GAN Model:
     * The GAN model combines the generator and discriminator networks. The generator generates fake images, which are then fed into the discriminator along with real images. The GAN model is trained to optimize the performance of both networks. In the code, the `gan(dis, gen)` function defines the GAN model.

## :computer: Environment :computer:
* This project is interpreted/tested on Ubuntu 20.04 LTS using python3


## :white_check_mark: Requirements :white_check_mark:
* numpy
* keras
* matplotlib


## Code Structure
* The provided code is structured as follows:

     * Importing the necessary libraries and modules.
     * Loading the dataset using the load_data_set() function from the preprocess module.
     * Initializing a new run using the Weights & Biases library (wandb).
     * Defining the architecture of the discriminator network using the discriminator() function.
     * Defining the architecture of the generator network using the generator(n) function.
     * Combining the discriminator and generator networks to create the GAN model using the gan(dis, gen) function.
     * Training the GAN model for a specified number of epochs.
     * Printing the discriminator and generator loss after each epoch.
     * Generating and displaying a grid of images using the trained generator.


## Code Examples
* Here are some code examples to illustrate the usage of the functions in the provided code:

     * Creating the discriminator network:
     `discrim = discriminator()`
     * Creating the generator network:
     `geney = generator(100)`
     * Creating the GAN model:
     `gan_model = gan(discrim, geney)`
     * Training the GAN model:
     `gan_model.train_on_batch(noise, valid_y)`



## :robot: The Experiments :robot:
* EXP1
     * Two Batch Normalization layers added to generator model.
* EXP2
     * Changed the learning rate of the GAN model from 0.0002 to 0.0001.
* EXP3
     * Changed the learning rate of the GAN model back to 0.0002.
     * Changed the precision tuning from 'float32' to 'float16'


## :memo: Authors :memo:
- Julia Bullard - [Github](https://github.com/Julia-5534)


## :scroll: License :scroll:
- TBA

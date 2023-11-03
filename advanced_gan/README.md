<p align="center">
  <Deep Convolutional Generative Adversarial Networks>
</p>


<p align="center">
<br>A Good Ol' Advanced DCGAN&trade; By</br>
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
     * GANs are a type of neural network architecture that consists of two main components - a generator and a discriminator. The generator generates fake images, while the discriminator tries to distinguish between real and fake images. The two components are trained simultaneously, with the goal of the generator generating images that are indistinguishable from real images.

* Deep Convolutional Neural Networks (DCNNs):
     * DCNNs are a type of neural network architecture that are specifically designed for image processing tasks. They consist of multiple layers of convolutional and pooling operations, which allow them to learn hierarchical representations of images.

* CIFAR-10 Dataset:
     * The CIFAR-10 dataset is a widely used benchmark dataset for image classification tasks. It consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. The dataset is commonly used for training and evaluating deep learning models.

## :computer: Environment :computer:
* This project is interpreted/tested on Ubuntu 20.04 LTS using python3


## :white_check_mark: Requirements :white_check_mark:
* numpy
* keras
* matplotlib
* tqdm
* os
* wandb


## Code Structure
* The provided code is structured as follows:

     * Importing the necessary libraries and modules:
          * The code begins by importing the required libraries and modules, including numpy, matplotlib, keras, tqdm, and wandb.

     * Loading and preprocessing the CIFAR-10 dataset:
          * The code then loads the CIFAR-10 dataset using the cifar10.load_data() function from the keras.datasets module. The dataset is then normalized to the range [-1, 1] by dividing each pixel value by 127.5 and subtracting 1.

     * Defining the discriminator model:
          * The code defines the discriminator model using the Sequential API from Keras. The discriminator consists of convolutional layers with leaky ReLU activation functions, followed by a flatten layer and a dense layer with a sigmoid activation function.

     * Defining the generator model:
          * The code defines the generator model using the Sequential API from Keras. The generator consists of dense layers with batch normalization and ReLU activation functions, followed by reshape and transposed convolutional layers with batch normalization and ReLU activation functions.

     * Defining the GAN model:
          * The code defines the GAN model by combining the generator and discriminator models. The discriminator is set to be non-trainable during the training of the GAN model.

     * Training the GAN model:
          * The code then trains the GAN model for a specified number of epochs. In each epoch, it iterates over the training data in batches, generates fake images using the generator, and trains the discriminator and generator models using the generated and real images. The losses of the discriminator and generator are logged and visualized using the wandb library.

     * Generating and saving sample images:
          * After each epoch, the code generates sample images using the generator and saves them for visualization.


## Code Examples
* Here are some code examples to illustrate the usage of the functions in the provided code:

`epochs = 50`
`batch_size = 128`

`for epoch in range(epochs):`
    `for batch in tqdm_notebook(range(len(X) // batch_size)):`
        `# Generate random noise`
        `noise = np.random.normal(0, 1, [batch_size // 2, 100])`

        `# Generate fake images`
        `fake_images = generator_model.predict(noise)`

        `# Train discriminator on real and fake data separately`
        `real_images = X[np.random.randint(0, len(X), batch_size // 2)]`
        `discriminator_loss_real = discriminator_model.train_on_batch(real_images, np.ones((batch_size // 2, 1)))`
        `discriminator_loss_fake = discriminator_model.train_on_batch(fake_images, np.zeros((batch_size // 2, 1)))`

        `# Calculate the total discriminator loss`
        `discriminator_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_fake)`

        `# Generate new noise for the generator`
        `noise = np.random.normal(0, 1, [batch_size, 100])`

        `# Train the generator within the GAN model`
        `generator_loss = gan_model.train_on_batch(noise, np.ones((batch_size, 1)))`

    `print("Epoch:", epoch)`
    `print("Discriminator Loss:", discriminator_loss)`
    `print("Generator Loss:", generator_loss)`

## Baseline
     * For this Advanced DCGAN, I decided to use the CIFAR-10 dataset.
     * Data loading and preprocessing are adjusted for CIFAR-10.
     * The input shape of the discriminator and generator is modified to accommodate 32x32 color images.
     * The number of output channels in the generator's final Conv2DTranspose layer is set to 3 for RGB images.
     * The generated images are rescaled from the [-1, 1] range to [0, 1] for display.
     *This code should work for training a DCGAN on the CIFAR-10 dataset. You can adjust the hyperparameters as needed and monitor the training progress using Weights & Biases (wandb).

## :robot: The Experiments :robot:
* EXP1
     * 
* EXP2
     * 


## :memo: Authors :memo:
- Julia Bullard - [Github](https://github.com/Julia-5534)


## :scroll: License :scroll:
- TBA

import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Input, LeakyReLU, Reshape
from keras.models import Model, Sequential
from keras.optimizers import Adam
from tqdm import tqdm_notebook
from data.adv_preprocess import load_cartoon_set
import wandb

X = load_cartoon_set()

# Initialize a new wandb run
wandb.init(project="exp1-adv-gan-cartoon")

# Adjust the discriminator
def discriminator():
    input = Input(shape=(64, 64, 3))  # Adjust for size and color channels of Cartoon Set
    
    x = Flatten()(input)
    x = Dropout(0.4)(x)
    x = Dense(1024, activation=LeakyReLU(alpha=0.2))(x)
    x = Dropout(0.4)(x)
    x = Dense(512, activation=LeakyReLU(alpha=0.2))(x)
    x = Dropout(0.4)(x)
    x = Dense(512, activation=LeakyReLU(alpha=0.2))(x)

    output = Dense(1, activation='sigmoid')(x)  # Output is a single value

    model = Model(input, output)
    model.compile(optimizer=Adam(lr=0.0002, beta_1=0.5), loss='binary_crossentropy')

    return model


# Adjust the generator
def generator(n):
    input = Input(shape=(n))

    x = Dense(256, activation=LeakyReLU(alpha=0.2))(input)
    x = Dense(512, activation=LeakyReLU(alpha=0.2))(x)
    x = Dense(1024, activation=LeakyReLU(alpha=0.2))(x)
    x = Dense(64*64*3, activation='tanh')(x)  # Adjust for size and color channels of Cartoon Set
    output = Reshape((64, 64, 3))(x)  # Adjust for size and color channels of Cartoon Set
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

# Adjust hyperparameters
epochs = 200  # Increase number of epochs
batch_size = 512  # Increase batch size
half_batch = batch_size // 2
n = 100

for i in range(epochs):
    print("EPOCH", i)
    for j in tqdm_notebook(range(len(X) // batch_size)):
        x_real, y_real = X[np.random.randint(0, len(X), half_batch)].reshape(half_batch, 64, 64, 3), np.ones(half_batch).reshape(half_batch, 1)
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
                img = geney.predict(np.random.randn(1 * n).reshape(1, n)).reshape(64, 64)  # Adjust for size of Cartoon Set
                axes[ii, jj].imshow(img)
                images.append(wandb.Image(img))  # Log image to W&B
        plt.show()
        plt.close()
        wandb.log({"Generated Images": images})  # Log all images to W&B

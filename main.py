from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
tf.enable_eager_execution()

import math
import glob
import imageio as io
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import time
import pickle

from IPython import display

dataset = []
r_dataset = []
g_dataset = []
b_dataset = []

BATCH_SIZE = 256
generator_input_size = 3

def getColorChannel(arr, index):

    result = []
    for y in arr:
        for x in y:
            channel = x
            color = channel[index]
            result.append(color)

    return result


# all images are 200 x 200
dirs = os.listdir('images')
dirs_length = len(dirs)
for file in dirs:

    pixel_data = io.imread('images/' + file)
    r = getColorChannel(pixel_data, 0)
    g = getColorChannel(pixel_data, 1)
    b = getColorChannel(pixel_data, 2)
    r_dataset.append(r)
    g_dataset.append(g)
    b_dataset.append(b)

    labels = file.split('_')
    obj = {
        'age': labels[0],
        'gender': labels[1],
        'race': labels[2]
    }
    dataset.append(obj)

    index = dirs.index(file)
    if index % 50 == 0:
        print(f"{index} out of {dirs_length} file loaded")

print('dataset loaded')

with open('red_pixel_dataset.txt', 'wb') as fp:
    pickle.dump(r_dataset, fp)

with open('green_pixel_dataset.txt', 'wb') as fp:
    pickle.dump(g_dataset, fp)

with open('blue_pixel_dataset.txt', 'wb') as fp:
    pickle.dump(b_dataset, fp)

print('dataset saved')

# age: 1-oldest
# gender: 0 (male), 1 (female)
# race: 0 (while), 1 (black), 2 (Asian), 3 (Indian), 4 (Other)

def make_generator_model():

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(10 * 10 * 256, use_bias=False, input_shape=(generator_input_size,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Reshape((10, 10, 256)))
    assert model.output_shape == (None, 10, 10, 256)  # Note: None is the batch size

    model.add(tf.keras.layers.Conv2DTranspose(filters=128,
                                              kernel_size=(5, 5),
                                              strides=(2, 2),
                                              padding='same',
                                              use_bias=False))
    assert model.output_shape == (None, 20, 20, 128)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(filters=64,
                                              kernel_size=(5, 5),
                                              strides=(2, 2),
                                              padding='same',
                                              use_bias=False))
    assert model.output_shape == (None, 40, 40, 64)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(
        tf.keras.layers.Conv2DTranspose(filters=1,
                                        kernel_size=(5, 5),
                                        strides=(5, 5),
                                        padding='same',
                                        use_bias=False,
                                        activation='tanh'))
    assert model.output_shape == (None, 200, 200, 1)

    return model

def make_discriminator_model():

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=64,
                                     kernel_size=(5, 5),
                                     strides=(2, 2),
                                     padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.MaxPool2D(pool_size=[3, 3],
                                        strides=3))

    model.add(tf.keras.layers.Conv2D(filters=128,
                                     kernel_size=(5, 5),
                                     strides=(2, 2),
                                     padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.MaxPool2D(pool_size=[3, 3],
                                        strides=3))

    model.add(tf.keras.layers.Conv2D(filters=128,
                                     kernel_size=(5, 5),
                                     strides=(2, 2),
                                     padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.MaxPool2D(pool_size=[2, 2],
                                        strides=2))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))

    return model

r_generator = make_generator_model()
r_discriminator = make_discriminator_model()

g_generator = make_generator_model()
g_discriminator = make_discriminator_model()

b_generator = make_generator_model()
b_discriminator = make_discriminator_model()

print('models made')

def generator_loss(generated_output):
    return tf.losses.sigmoid_cross_entropy(tf.ones_like(generated_output), generated_output)


def discriminator_loss(real_output, generated_output):

    # [1,1,...,1] with real output since it is true and
    # we want our generated examples to look like it
    real_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(real_output), logits=real_output)

    # [0,0,...,0] with generated images since they are fake
    generated_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(generated_output), logits=generated_output)

    total_loss = real_loss + generated_loss

    return total_loss

r_generator_optimizer = tf.train.AdamOptimizer(1e-4)
r_discriminator_optimizer = tf.train.AdamOptimizer(1e-4)

g_generator_optimizer = tf.train.AdamOptimizer(1e-4)
g_discriminator_optimizer = tf.train.AdamOptimizer(1e-4)

b_generator_optimizer = tf.train.AdamOptimizer(1e-4)
b_discriminator_optimizer = tf.train.AdamOptimizer(1e-4)

print('optimerizers added')

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(r_generator_optimizer=r_generator_optimizer,
                                 r_discriminator_optimizer=r_discriminator_optimizer,
                                 g_generator_optimizer=g_generator_optimizer,
                                 g_discriminator_optimizer=g_discriminator_optimizer,
                                 b_generator_optimizer=b_generator_optimizer,
                                 b_discriminator_optimizer=b_discriminator_optimizer,

                                 r_generator=r_generator,
                                 r_discriminator=r_discriminator,
                                 g_generator=g_generator,
                                 g_discriminator=g_discriminator,
                                 b_generator=b_generator,
                                 b_discriminator=b_discriminator)

EPOCHS = 50
num_examples_to_generate = 36

random_vector_for_generation = tf.random_normal([num_examples_to_generate,
                                                 generator_input_size])

def train_step(r_images, g_images, b_images):

    # generating noise from a normal distribution
    noise = tf.random_normal([BATCH_SIZE, generator_input_size])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        r_generated_images = r_generator(noise, training=True)
        g_generated_images = g_generator(noise, training=True)
        b_generated_images = b_generator(noise, training=True)

        r_real_output = r_discriminator(r_images, training=True)
        g_real_output = g_discriminator(g_images, training=True)
        b_real_output = b_discriminator(b_images, training=True)

        r_generated_output = r_discriminator(r_generated_images, training=True)
        g_generated_output = g_discriminator(g_generated_images, training=True)
        b_generated_output = b_discriminator(b_generated_images, training=True)

        r_gen_loss = generator_loss(r_generated_output)
        g_gen_loss = generator_loss(g_generated_output)
        b_gen_loss = generator_loss(b_generated_output)

        r_disc_loss = discriminator_loss(r_real_output, r_generated_output)
        g_disc_loss = discriminator_loss(g_real_output, g_generated_output)
        b_disc_loss = discriminator_loss(b_real_output, b_generated_output)

    r_gradients_of_generator = gen_tape.gradient(r_gen_loss, r_generator.variables)
    g_gradients_of_generator = gen_tape.gradient(g_gen_loss, g_generator.variables)
    b_gradients_of_generator = gen_tape.gradient(b_gen_loss, b_generator.variables)

    r_gradients_of_discriminator = disc_tape.gradient(r_disc_loss, r_discriminator.variables)
    g_gradients_of_discriminator = disc_tape.gradient(g_disc_loss, g_discriminator.variables)
    b_gradients_of_discriminator = disc_tape.gradient(b_disc_loss, b_discriminator.variables)

    r_generator_optimizer.apply_gradients(zip(r_gradients_of_generator, r_generator.variables))
    g_generator_optimizer.apply_gradients(zip(g_gradients_of_generator, g_generator.variables))
    b_generator_optimizer.apply_gradients(zip(b_gradients_of_generator, b_generator.variables))

    r_discriminator_optimizer.apply_gradients(zip(r_gradients_of_discriminator, r_discriminator.variables))
    g_discriminator_optimizer.apply_gradients(zip(g_gradients_of_discriminator, g_discriminator.variables))
    b_discriminator_optimizer.apply_gradients(zip(b_gradients_of_discriminator, b_discriminator.variables))

train_step = tf.contrib.eager.defun(train_step)

def train(epochs):

    for epoch in range(epochs):
        start = time.time()

        for index in range(len(r_dataset)):
            train_step(r_dataset[index],
                       g_dataset[index],
                       b_dataset[index])

        display.clear_output(wait=True)
        generate_and_save_images([r_generator, g_generator, b_generator],
                                 epoch + 1,
                                 random_vector_for_generation)

        # saving (checkpoint) the model every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time taken for epoch {} is {} sec'.format(epoch + 1,
                                                         time.time() - start))
    # generating after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images([r_generator, g_generator, b_generator],
                             epochs,
                             random_vector_for_generation)

def generate_and_save_images(models, epoch, test_input):

    # make sure the training parameter is set to False because we
    # don't want to train the batchnorm layer when doing inference.
    predictions = []
    for i in range(len(models)):
        prediction = models[i](test_input, training=False)
        predictions.append(prediction)

    pixels = []
    for i in range(len(predictions[0])):

        # get x, y cordinate of pixel
        x = i % 200
        y = i / 200 - x

        r = predictions[0][y][x]
        g = predictions[1][y][x]
        b = predictions[2][y][x]

        # if x = 0; create a new row
        if x == 0:
            pixels.append([])

        pixels[y].append([r, g, b])

    dimensions = math.floor(math.sqrt(num_examples_to_generate))
    fig = plt.figure(figsize=(dimensions, dimensions))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(pixels)
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

print('training started')
train(EPOCHS)
# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

def display_image(epoch_no):
    return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

display_image(EPOCHS)

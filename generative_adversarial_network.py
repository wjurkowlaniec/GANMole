"""
This code enables you to generate melanoma images using a Generative Adversarial Network (GAN)
@author: Abder-Rahman Ali
abder@cs.stir.ac.uk
"""

import keras
from keras import layers
import numpy as np
import cv2
import os
import sys
import random  # Added import random
from keras.preprocessing import image

latent_dimension = 32
height = 600
width = 600
channels = 3
iterations = 100

BATCH_SIZE = 70
real_images = []

# paths to the training and results directories
TRAIN_DIRECTORY = "train"
RESULTS_DIRECTORY = "results"

# GAN generator for 600x600
start_size = 15  # Starting spatial dimension
start_channels = 512  # Starting channels

generator_input = keras.Input(shape=(latent_dimension,))

# Project latent vector to initial feature map
x = layers.Dense(start_size * start_size * start_channels)(generator_input)
x = layers.Reshape((start_size, start_size, start_channels))(x)

# Upsampling blocks (Conv2DTranspose -> BatchNormalization -> LeakyReLU)
# 15x15 -> 30x30
x = layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(negative_slope=0.2)(x)

# 30x30 -> 60x60
x = layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(negative_slope=0.2)(x)

# 60x60 -> 120x120
x = layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(negative_slope=0.2)(x)

# 120x120 -> 600x600
x = layers.Conv2DTranspose(
    channels, kernel_size=5, strides=5, padding="same", activation="tanh"
)(x)

generator = keras.models.Model(generator_input, x, name="generator")
generator.summary()

# GAN discriminator for 600x600
discriminator_input = layers.Input(shape=(height, width, channels))

# Downsampling blocks (Conv2D -> BatchNormalization -> LeakyReLU)
# 600x600 -> 120x120
x = layers.Conv2D(64, kernel_size=5, strides=5, padding="same")(discriminator_input)
# No BatchNormalization on the first conv layer is a common practice
x = layers.LeakyReLU(negative_slope=0.2)(x)

# 120x120 -> 60x60
x = layers.Conv2D(128, kernel_size=4, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(negative_slope=0.2)(x)

# 60x60 -> 30x30
x = layers.Conv2D(256, kernel_size=4, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(negative_slope=0.2)(x)

# 30x30 -> 15x15
x = layers.Conv2D(512, kernel_size=4, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(negative_slope=0.2)(x)

x = layers.Flatten()(x)

# dropout layer
x = layers.Dropout(0.4)(x)

# classification layer
x = layers.Dense(1, activation="sigmoid")(x)

discriminator = keras.models.Model(discriminator_input, x, name="discriminator")
discriminator.summary()

discriminator_optimizer = keras.optimizers.RMSprop(
    learning_rate=0.0008, clipvalue=1.0, decay=1e-8
)

discriminator.compile(optimizer=discriminator_optimizer, loss="binary_crossentropy")

# adversarial network
discriminator.trainable = False

gan_input = keras.Input(shape=(latent_dimension,))
gan_output = discriminator(generator(gan_input))
gan = keras.models.Model(gan_input, gan_output)

gan_optimizer = keras.optimizers.RMSprop(
    learning_rate=0.0004, clipvalue=1.0, decay=1e-8
)

gan.compile(optimizer=gan_optimizer, loss="binary_crossentropy")

for step in range(iterations):
    # create random vectors as input for the generator (use batch_size)
    random_latent_vectors = np.random.normal(size=(BATCH_SIZE, latent_dimension))
    # decode the random latent vectors into fake images
    generated_images = generator.predict(random_latent_vectors)

    real_images_list = []
    # Use train_directory defined earlier
    print(f"Loading {BATCH_SIZE} real images from {TRAIN_DIRECTORY}...")
    try:
        image_files = [
            f
            for f in os.listdir(TRAIN_DIRECTORY)
            if os.path.isfile(os.path.join(TRAIN_DIRECTORY, f))
            and f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if len(image_files) < BATCH_SIZE:
            print(
                f"Error: Not enough images in {TRAIN_DIRECTORY} ({len(image_files)}) to form a batch of size {BATCH_SIZE}."
            )
            sys.exit(1)
        # Randomly sample BATCH_SIZE filenames
        sampled_files = random.sample(image_files, BATCH_SIZE)
    except FileNotFoundError:
        print(f"Error: Training directory '{TRAIN_DIRECTORY}' not found.")
        sys.exit(1)

    for filename in sampled_files:
        img_path = os.path.join(TRAIN_DIRECTORY, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(
                f"Error: Failed to load image {img_path}. Please ensure it exists and is readable."
            )
            # Optionally skip this image and try another, or exit
            sys.exit(1)  # Exit if an image is missing/unreadable
        # Resize image
        img_resized = cv2.resize(img, (width, height))
        # Preprocess: Convert to float32 and scale to [-1, 1] (matching generator's tanh output)
        img_processed = (img_resized.astype("float32") / 255.0) * 2.0 - 1.0
        real_images_list.append(img_processed)

    # convert real_images list to numpy array (should now be 4D)
    real_images = np.array(real_images_list)
    print(
        f"Generated images shape: {generated_images.shape}, dtype: {generated_images.dtype}"
    )
    print(f"Real images shape: {real_images.shape}, dtype: {real_images.dtype}")
    # Dtype conversion is now handled during preprocessing, ensure generated images are also float32 (should be by default)
    print(
        f"Real images shape after preprocessing: {real_images.shape}, dtype: {real_images.dtype}"
    )

    # Check shapes before concatenating
    if (
        generated_images.shape[1:] != real_images.shape[1:]
        or generated_images.shape[0] != real_images.shape[0]
    ):
        print(f"Error: Shape mismatch before concatenation!")
        print(f"Generated shape: {generated_images.shape}")
        print(f"Real shape: {real_images.shape}")
        sys.exit(1)

    # combine fake images with real images
    combined_images = np.concatenate([generated_images, real_images])
    print(f"Combined images shape: {combined_images.shape}")
    # assemble labels: 1s for real images, 0s for fake images (or vice versa depending on convention)
    # The combined_images has shape (2 * batch_size, ...), so labels must match.
    # Original code might have label order reversed (ones for fake, zeros for real) - adjusting to common practice (ones=real, zeros=fake)
    # But generated are first, real are second in combined_images.
    labels = np.concatenate([np.zeros((BATCH_SIZE, 1)), np.ones((BATCH_SIZE, 1))])
    print(f"Labels shape: {labels.shape}")
    # add random noise to the labels (label smoothing)
    labels = labels + 0.05 * np.random.random(labels.shape)
    # train the discriminator
    discriminator_loss = discriminator.train_on_batch(combined_images, labels)

    # Train the generator (via the combined GAN model where discriminator weights are frozen)
    # Create a new batch of random latent vectors (use BATCH_SIZE)
    random_latent_vectors_for_gan = np.random.normal(
        size=(BATCH_SIZE, latent_dimension)
    )
    # Assemble labels that classify the fake images as "real" (all ones)
    misleading_targets = np.ones((BATCH_SIZE, 1))
    # train the generator via the GAN model
    # Note: Discriminator weights are frozen in the 'gan' model definition, so only generator weights are updated here.
    adversarial_loss = gan.train_on_batch(
        random_latent_vectors_for_gan, misleading_targets
    )
    gan.save_weights("gan.weights.h5")
    print("discriminator loss: ")
    print(discriminator_loss)
    print("adversarial loss: ")
    print(adversarial_loss)

    # Save the first generated image every 50 steps
    if step % 50 == 0:
        # Save generated images (scaling back from [-1, 1] to [0, 255])
        if generated_images.shape[0] > 0: # Check if images were generated
            i = 0 # Save only the first image (index 0)
            # Scale from [-1, 1] to [0, 1], then to [0, 255]
            img_array_scaled = ((generated_images[i] + 1) / 2.0 * 255.0)
            # Ensure it's uint8
            img_array_uint8 = img_array_scaled.astype(np.uint8)
            img = image.array_to_img(img_array_uint8)
            save_path = os.path.join(
                RESULTS_DIRECTORY, f"generated_image_step{step}.png" # Simpler filename
            )
            # Check if the results directory exists, if not, create it
            if not os.path.exists(RESULTS_DIRECTORY):
                os.makedirs(RESULTS_DIRECTORY)
            img.save(save_path)
            print(f"Saved generated image sample for step {step}")

    # Note: real_images is reloaded each step, no need to clear it here.

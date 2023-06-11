import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json

def build_autoencoder(image, code_size):
    #Function input: image and size of encoding
    #Function output: encoder and decoder
    #Get the shape of the input image
    img_shape = image.shape
    
    # Encoder architecture
    encoder = tf.keras.Sequential([
        tf.keras.layers.InputLayer(img_shape),
        # Convolutional layer with 64 filters of size 3x3 and ReLU activation
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        # Max pooling layer with pool size of 2x2 and stride of 2
        tf.keras.layers.MaxPool2D((2, 2), padding='same'),
        tf.keras.layers.Dropout(0.3),

        # Convolutional layer with 32 filters of size 3x3 and ReLU activation
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        # Max pooling layer with pool size of 2x2 and stride of 2
        tf.keras.layers.MaxPool2D((2, 2), padding='same'),
        tf.keras.layers.Dropout(0.2),

        # Convolutional layer with 16 filters of size 3x3 and ReLU activation
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        # Max pooling layer with pool size of 2x2 and stride of 2
        tf.keras.layers.MaxPool2D((2, 2), padding='same'),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(code_size)
    ])

    # Decoder architecture
    decoder = tf.keras.Sequential([
        tf.keras.layers.InputLayer((code_size,)),
        tf.keras.layers.Dense(np.prod(img_shape)),
        tf.keras.layers.Reshape(img_shape),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(1, (2, 2), activation='relu', padding='same')
    ])

    return encoder, decoder


def image2grayscale(image):
    # Convert RGB image to grayscale
    R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    imgGray = 0.2989 * R + 0.5870 * G + 0.1140 * B

    # Normalize the grayscale image to the range [0, 1]
    imgGray = imgGray.astype('float32') / 255.

    # Reshape the image
    imgGray = imgGray.reshape((len(imgGray), np.prod(imgGray.shape[1:])))
    return imgGray


def show_image(x):
    # Display an image
    plt.imshow(np.clip(x, 0, 255))


def visualize(img, encoder, decoder):
    """Draws original, encoded, and decoded images"""
    # Encode the input image
    code = encoder.predict(img[None])[0]
    # Decode the encoded image
    reco = decoder.predict(code[None])[0]

    # Display the original image
    plt.subplot(1, 3, 1)
    plt.title("Original")
    show_image(img)

    # Display the encoded image
    plt.subplot(1, 3, 2)
    plt.title("Code")
    plt.imshow(code.reshape([code.shape[-1] // 2, -1]))

    




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





    




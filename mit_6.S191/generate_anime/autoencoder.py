import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def show_image(x):
    # Display an image
    plt.imshow(np.clip(x, 0, 255))




class Autoencoder:
    def __init__(self, image, code_size):
        super(Autoencoder, self).__init__()
        self.img = image
        self.image_shape = image.shape
        self.code_size = code_size
        self.build_autoencoder()

    def build_autoencoder(self):
        # Encoder architecture
        encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(self.image_shape),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPool2D((2, 2), padding='same'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPool2D((2, 2), padding='same'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPool2D((2, 2), padding='same'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.code_size)
        ])

        # Decoder architecture
        decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer((self.code_size,)),
            tf.keras.layers.Dense(np.prod(self.image_shape)),
            tf.keras.layers.Reshape(self.image_shape),
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(1, (2, 2), activation='relu', padding='same')
        ])
        self.encoder, self.decoder = encoder, decoder
        return self

def visualize(layer : Autoencoder):
    """Draws original, encoded, and decoded images"""
    # Encode the input image
    code = layer.encoder.predict(layer.img[None])[0]
    # Decode the encoded image
    reco = layer.decoder.predict(code[None])[0]

    # Display the original image
    plt.subplot(1, 3, 1)
    plt.title("Original")
    show_image(layer.img)

    # Display the encoded image
    plt.subplot(1, 3, 2)
    plt.title("Code")
    plt.imshow(code.reshape([code.shape[-1] // 2, -1]))

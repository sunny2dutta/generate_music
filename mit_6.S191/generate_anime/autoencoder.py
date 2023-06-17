import tensorflow as tf
import numpy as np

class Autoencoder:
    def __init__(self, image, code_size):
        super(Autoencoder, self).__init__()
        self.image_shape = image.shape
        self.code_size = code_size
        self.encoder, self.decoder = self.build_autoencoder()

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

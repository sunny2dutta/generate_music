import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf

dataset_path = '/Users/debaryadutta/learn_dl/mit_6.S191/data_anime/'
arr = os.listdir(dataset_path)

path = dataset_path+arr[0]
image = plt.imread(path)

input_shape = image.shape

latent_dim = 512

# Load and preprocess the face images
def load_images(path):
    images = []
    for filename in os.listdir(path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img = Image.open(os.path.join(path, filename))
            img = img.resize(input_shape[:2])
            img = np.array(img) / 255.0
            images.append(img)
    return np.array(images)

# Encoder network
def build_encoder(input_shape, latent_dim):
    inputs = tf.keras.Input(shape=input_shape)
    
    encoder = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape),
        # Convolutional layer with 64 filters of size 3x3 and ReLU activation
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        # Max pooling layer with pool size of 2x2 and stride of 2
        tf.keras.layers.MaxPool2D((2, 2),padding='same'),
        tf.keras.layers.Dropout(0.3),

        # Convolutional layer with 128 filters of size 3x3 and ReLU activation
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        # Max pooling layer with pool size of 2x2 and stride of 2
        tf.keras.layers.MaxPool2D((2,2),padding='same'),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        # Max pooling layer with pool size of 2x2 and stride of 2
        tf.keras.layers.MaxPool2D((2,2),padding='same'),

        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(latent_dim)
    ])
    z_mean = encoder(inputs)
    z_log_var = encoder(inputs)
    return tf.keras.Model(inputs, [z_mean, z_log_var], name='encoder')

# Decoder network
def build_decoder(latent_dim,input_shape):
    
    latent_inputs = tf.keras.layers.Input((latent_dim,))

    outputs=tf.keras.Sequential([
            tf.keras.layers.InputLayer((latent_dim,)),
            tf.keras.layers.Dense(np.prod(input_shape)),
            tf.keras.layers.Reshape(input_shape),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(3, (22, 22), activation='relu', padding='same')
            ])(latent_inputs)

    return tf.keras.Model(latent_inputs, outputs, name='decoder')

# VAE model




def build_vae(encoder, decoder):
    inputs = encoder.input
    z_mean, z_log_var = encoder(inputs)
    z = tf.keras.layers.Lambda(sample_latent)([z_mean, z_log_var])
    reconstructed = decoder(z)
    model = tf.keras.Model(inputs, reconstructed, name='vae')
    
    # Compute the reconstruction loss
    reconstruction_loss = 0
    reconstruction_loss = tf.keras.backend.mean(tf.keras.losses.mean_squared_error(inputs, reconstructed))
    #reconstruction_loss *= input_shape[0] * input_shape[1] * input_shape[2]  # Scale the loss
    
    # Compute the KL divergence loss
    kl_loss = -0.5 * tf.keras.backend.mean(1 + z_log_var - tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_log_var), axis=-1)
    
    # Combine the losses
    total_loss = reconstruction_loss + kl_loss
    
    model.add_loss(total_loss)

    #ßtf.keras.utils.plot_model(model, "model.png", show_shapes=True, )

    return model



# Custom sampling function for the latent space
def sample_latent(args):
    z_mean, z_log_var = args
    batch_size = tf.keras.backend.shape(z_mean)[0]
    latent_dim = tf.keras.backend.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch_size, latent_dim))
    return z_mean + tf.keras.backend.exp(0.5 * z_log_var) * epsilon

import autoencoder as autoencoder
import preprocess as pp
import tensorflow as tf
import matplotlib.pyplot as plt
import  visualisation as vs
import anime_vae as vae
import os
import numpy as np
dataset_path = '/Users/debaryadutta/learn_dl/mit_6.S191/data_anime/'

images_data = vae.load_images(dataset_path)

vs.show_image_dir(dataset_path,5,4)

arr = os.listdir(dataset_path)

path = dataset_path+arr[0]
image = plt.imread(path)

input_shape = image.shape

print(input_shape)

latent_dim = 32

encoder = vae.build_encoder(input_shape, latent_dim)

decoder = vae.build_decoder(latent_dim,input_shape)

vae_model = vae.build_vae(encoder, decoder)


vae_model.compile(optimizer='adam')
#vae_model.fit(x_train, epochs=500, batch_size=batch_size, validation_data=(x_test, None))



num_samples = 10
latent_vectors = np.random.normal(size=(num_samples, latent_dim))

# Decode the latent vectors to generate images
generated_images = decoder.predict(latent_vectors)

generated_images = generated_images.reshape((-1, 60, 60))

# Plot the generated images
fig, axes = plt.subplots(1, num_samples, figsize=(10, 2))
for i in range(num_samples):
    axes[i].imshow(generated_images[i].reshape(60,60), cmap='gray')
    axes[i].axis('off')
plt.tight_layout()
plt.show()

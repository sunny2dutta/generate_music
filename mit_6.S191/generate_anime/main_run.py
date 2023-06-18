import autoencoder as autoencoder
import preprocess as pp
import tensorflow as tf
import matplotlib.pyplot as plt

image_path = '/Users/debaryadutta/learn_dl/mit_6.S191/data_set_images/digit2.jpg'
code_size = 32

image = plt.imread(image_path)
plt.imshow(image)
plt.title('Input image')
plt.show()

autoencoder_layer = autoencoder.Autoencoder(image, code_size)

autoencoder.visualize(autoencoder_layer)
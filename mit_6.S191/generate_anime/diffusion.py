import numpy as np
import matplotlib.pyplot as plt
import preprocess as pp
import tensorflow as tf
import matplotlib.pyplot as plt
import  visualisation as vs
import anime_vae as vae
import os
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter


def forward_diffusion_image(image, drift, volatility, time_step):
    rows, cols, _ = image.shape
    random_noise = np.random.normal(0, 1, (rows, cols, 3))
    delta_image = drift * image * time_step + volatility * image * np.sqrt(time_step) * random_noise
    new_image = image + delta_image
    return new_image


dataset_path = '/Users/debaryadutta/learn_dl/mit_6.S191/data_anime/'
drift = 0.05
volatility = 0.2
time_step = 0.01


images_data = vae.load_images(dataset_path)

vs.show_image_dir(dataset_path,5,4)

arr = os.listdir(dataset_path)

path = dataset_path+arr[0]
image = plt.imread(path)

new_image = forward_diffusion_image(image, drift, volatility, time_step)

# Plotting

arr = os.listdir(dataset_path)

fig, axs = plt.subplots(5, 4, figsize=(10, 4))
for i, ax in enumerate(axs.flat):
    path = dataset_path + arr[i]
    image = plt.imread(path)
    new_image = forward_diffusion_image(image, drift, volatility, time_step)

    img_clipped = np.clip(new_image, 0, 255)

    ax.imshow(img_clipped)
    ax.axis('off')
plt.show()

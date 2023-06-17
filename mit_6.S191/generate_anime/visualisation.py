
import matplotlib.pyplot as plt




def show_image_dir(dataset_path,row_of_images,col_of_images):
    """
    This function takes input the path of images stored and displays the images.
    The function has no return value

    For displaying the function also takes input display stype of images
    """
    arr = os.listdir(dataset_path)

    fig, axs = plt.subplots(row_of_images,col_of_images, figsize=(10, 4))
    for i, ax in enumerate(axs.flat):
        path = dataset_path+arr[i]
        image = plt.imread(path)

        ax.imshow(image)
        ax.axis('off')
    plt.show()
    return 0




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


    # Display the encoded image
    plt.subplot(1, 3, 2)
    plt.title("Code")
    plt.imshow(code.reshape([code.shape[-1] // 2, -1]))

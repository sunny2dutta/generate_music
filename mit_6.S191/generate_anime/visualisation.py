
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

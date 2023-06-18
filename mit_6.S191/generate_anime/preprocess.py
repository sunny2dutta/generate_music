import cv2
import os




def convert_to_grayscale_dir(source_dir, file_name):
    """
    Convert RGB image to grayscale, when given an directory and filename
    """

    source_file = os.path.join(source_dir, file_name)
    gray_image = cv2.cvtColor(source_file, cv2.COLOR_BGR2GRAY)
    return gray_image

def convert_to_grayscale_dir(source_dir,destination_dir):
    """
    Convert all RGB image in input path to output directory
    """

    if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)

    file_list = os.listdir(source_dir)

    for file_name in file_list:
        # Get the full path of the source file
        source_file = os.path.join(source_dir, file_name)
        
        clr_image = cv2.imread(source_file)
        # Get the full path of the destination file
        file_name_gray = file_name
        gray_image = cv2.cvtColor(clr_image, cv2.COLOR_BGR2GRAY)

        destination_file = os.path.join(destination_dir, file_name_gray)
        cv2.imwrite(destination_file,gray_image)



    return 0






#source_dir = '/Users/debaryadutta/learn_dl/mit_6.S191/data_anime'

#destination_dir = '/Users/debaryadutta/learn_dl/mit_6.S191/data_anime_gray'

#convert_to_grayscale_dir(source_dir,destination_dir)
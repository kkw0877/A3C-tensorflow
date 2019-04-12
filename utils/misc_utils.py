import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize

def preprocess_image(image, height, width):
    """Change the color of the image from rgb to gray
       and then modify the size of image"""
    image = np.uint8(
        resize(rgb2gray(image), [height, width], mode='constant'))
    image = np.reshape(image, [height, width, 1])
    return image


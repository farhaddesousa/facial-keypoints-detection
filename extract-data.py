import numpy as np
import pandas as pd

def getTrainData(i):
    """ Retreives a single image's data from the training csv dataset.
    Args:
        i: index of the data point i you want to retrieve from training.csv

    Returns:
        data: a numpy array of size 96x96 containing the pixel greyscale values of the requested image

    """
    train_data = pd.read_csv('dataset/training.csv')

    data = np.fromstring(train_data.iloc[i, 30], dtype = int, sep = ' ')

    return data
    

data = getTrainData(2300)
print(data)
print(type(data))


#pip install pillow
from PIL import Image

def create_image_from_array(pixel_values):
    # Reshape the pixel values array into a 96x96 image
    pixel_array = np.array(pixel_values).reshape((96, 96))

    # Create a new image with the same size as the pixel array, in grayscale
    img = Image.new('L', (96, 96), "black")
    pixels = img.load()

    # Iterate through the pixel array and set pixel values accordingly
    for i in range(96):
        for j in range(96):
            # Set pixel value based on the correspond
            # ing value in the numpy array
            intensity = int(pixel_array[i, j])
            pixels[j, i] = intensity
    return img
# Example usage:
image = create_image_from_array(data)
image.save("trial_image.jpg")
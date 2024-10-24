### INGA CODE ###

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv("../dataset/training.csv")

def rotate_image_and_points(image_array, points, angle):
    """
    Rotates a 96x96 grayscale image and adjusts the facial feature locations accordingly.

    Parameters:
    - image_array: 96x96 numpy array representing the image.
    - points: A pandas Series with columns for (x, y) pairs of facial features.
    - angle: The rotation angle in degrees.

    Returns:
    - rotated_image: The rotated image (in uint8 format).
    - rotated_points: A numpy array with the rotated facial feature coordinates.
    """
    h, w = image_array.shape[:2]

    # Ensure the image is of type uint8 and pixel values are scaled correctly
    image_array = image_array.astype(np.uint8)

    # Compute the center of the image (rotation will be around the center)
    center = (w // 2, h // 2)

    # Compute the rotation matrix for the given angle
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Rotate the image using warpAffine
    rotated_image = cv2.warpAffine(image_array, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT)

    # Initialize a list to store rotated points
    rotated_points = []

    # Iterate over the (x, y) pairs in the facial feature points
    for i in range(0, len(points), 2):
        x = points[i]
        y = points[i + 1]

        # Apply the rotation matrix to each point
        point = np.array([[x], [y], [1]])  # Convert point to homogeneous coordinates
        rotated_point = np.dot(rotation_matrix, point).flatten()

        # Append the rotated x and y to the list
        rotated_points.extend([rotated_point[0], rotated_point[1]])

    return rotated_image, np.array(rotated_points)


def visualize_image_and_points(image_array, points, title):
    """
    Visualizes the image and plots the facial feature points.

    Parameters:
    - image_array: 96x96 grayscale image (numpy array)
    - points: List or array of facial feature points (x, y) pairs
    - title: Title of the plot
    """
    plt.figure(figsize=(5, 5))
    plt.imshow(image_array, cmap='gray')
    for i in range(0, len(points), 2):
        plt.scatter(points[i], points[i + 1], color='red')
    plt.title(title)
    plt.show()


# Example usage: Visualizing the rotated image and points
i = 0  # Replace with desired index from your dataset
image_data = np.fromstring(train_data.iloc[i]['Image'], dtype=int, sep=' ').reshape(96, 96)
points = train_data.iloc[i][:-1].to_numpy()

# Rotate the image and the points
rotated_image, rotated_points = rotate_image_and_points(image_data, points, 90)

# Visualize the original and rotated images
visualize_image_and_points(image_data, points, title="Original Image with Facial Points")
visualize_image_and_points(rotated_image, rotated_points, title="Rotated Image with Facial Points")

### END INGA CODE ###

# Rotate data frame

def create_rotated_dataframe(original_df):
    """
    Creates a new DataFrame with rotated images and corresponding facial feature locations.

    For each image in the original DataFrame, rotates the image and its facial feature points by a random angle between 0 and 360 degrees.

    Parameters:
    - original_df: The original DataFrame with facial feature locations and images.

    Returns:
    - rotated_df: A DataFrame containing the rotated images and facial feature points.
    """
    rotated_data = []

    for i in range(len(original_df)):
        # Extract the image and the points for each entry
        image_data = np.fromstring(original_df.iloc[i]['Image'], dtype=int, sep=' ').reshape(96, 96)
        points = original_df.iloc[i][:-1].to_numpy()  # Extract facial feature points (x, y) pairs

        # Generate a random angle between 0 and 360 degrees
        angle = np.random.uniform(-15, 15)

        # Rotate the image and points
        rotated_image, rotated_points = rotate_image_and_points(image_data, points, angle)

        # Convert the rotated image back to space-separated string format
        rotated_image_str = ' '.join(map(str, rotated_image.flatten()))

        # Combine the rotated points and image into one row, preserving the original structure
        rotated_row = np.concatenate([rotated_points, [rotated_image_str]])

        rotated_data.append(rotated_row)

    # Create a new DataFrame with the same columns as the original
    rotated_df = pd.DataFrame(rotated_data, columns=original_df.columns)

    return rotated_df


# Example: Rotate all images in the DataFrame by a random angle and create a new DataFrame
#rotated_df = create_rotated_dataframe(train_data)

# Step 1: Load the original dataset and drop NAs
train_csv_file_path = '../dataset/training.csv'
train_data = pd.read_csv(train_csv_file_path)
train_data = train_data.dropna()

# Step 2: Create two rotated datasets
rotated_df1 = create_rotated_dataframe(train_data)
rotated_df2 = create_rotated_dataframe(train_data)

# Step 3: Concatenate the original and rotated datasets
rotated_data = pd.concat([train_data, rotated_df1, rotated_df2], ignore_index=True)

# Shuffle the combined dataset
rotated_data = rotated_data.sample(frac=1).reset_index(drop=True)

# Step 4: Save the combined dataset to a new CSV file
rotated_data_path = '../dataset/rotated_training_constrained.csv'
rotated_data.to_csv(rotated_data_path, index=False)

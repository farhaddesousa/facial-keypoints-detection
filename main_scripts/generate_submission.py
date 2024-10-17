import numpy as np
import pandas as pd

# Load IdLookupTable and SampleSubmission format
id_lookup_table = pd.read_csv('dataset/IdLookupTable.csv')
sample_submission = pd.read_csv('dataset/SampleSubmission.csv')

# Load the predictions saved from your previous script
all_predictions = np.load('predictions_rotated.npy')  # Shape: (num_samples, 15, 2)

# Feature mapping (15 keypoints, x and y values)
keypoint_names = [
    'left_eye_center_x', 'left_eye_center_y',
    'right_eye_center_x', 'right_eye_center_y',
    'left_eye_inner_corner_x', 'left_eye_inner_corner_y',
    'left_eye_outer_corner_x', 'left_eye_outer_corner_y',
    'right_eye_inner_corner_x', 'right_eye_inner_corner_y',
    'right_eye_outer_corner_x', 'right_eye_outer_corner_y',
    'left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y',
    'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y',
    'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y',
    'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y',
    'nose_tip_x', 'nose_tip_y',
    'mouth_left_corner_x', 'mouth_left_corner_y',
    'mouth_right_corner_x', 'mouth_right_corner_y',
    'mouth_center_top_lip_x', 'mouth_center_top_lip_y',
    'mouth_center_bottom_lip_x', 'mouth_center_bottom_lip_y'
]

# Create a dictionary to map image IDs to their predicted keypoints
image_id_to_predictions = {}
for idx, keypoints in enumerate(all_predictions):
    image_id_to_predictions[idx + 1] = keypoints  # ImageId starts at 1

# Now populate the submission file using the IdLookupTable
submission_data = []

for index, row in id_lookup_table.iterrows():
    image_id = row['ImageId']
    feature_name = row['FeatureName']

    # Find the corresponding keypoint index
    keypoint_index = keypoint_names.index(feature_name)

    # Get the predicted keypoint value for the image and feature
    predicted_value = image_id_to_predictions[image_id].flatten()[keypoint_index]

    # Append the result to submission_data
    submission_data.append([row['RowId'], predicted_value])

# Create a DataFrame for the submission
submission_df = pd.DataFrame(submission_data, columns=['RowId', 'Location'])

# Save to CSV
submission_df.to_csv('submission_rotated.csv', index=False)

print("Submission file created successfully: submission_rotated.csv")

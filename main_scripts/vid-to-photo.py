import cv2
import os

# Path to the .mov video file
video_path = '../dataset/inga.mov'
output_dir = 'frames'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_path)

frame_count = 0

# Loop through frames in the video
while cap.isOpened():
    ret, frame = cap.read()

    # If a frame was successfully read
    if ret:
        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Resize to 96x96
        resized_frame = cv2.resize(gray_frame, (96, 96))

        # Save frame as an image
        frame_filename = os.path.join(output_dir, f'frame_{frame_count:04d}.png')
        cv2.imwrite(frame_filename, resized_frame)

        frame_count += 1
    else:
        break

# Release the video capture object
cap.release()
print(f'Total frames extracted: {frame_count}')

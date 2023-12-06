import os
import cv2

blur_radius = 0
alpha = 2.5
beta = -1.5

input_dir = 'data/test' # Input directory containing images and subdirectories
output_dir = 'equalised-unsharp-test-images/' # Output directory for sharpened images

os.makedirs(output_dir, exist_ok=True) # Create the output directory if it doesn't exist

for emotion_dir in os.listdir(input_dir): # Loop through each subdirectory for each emotion
    emotion_path = os.path.join(input_dir, emotion_dir)

    if os.path.isdir(emotion_path):
        for file in os.listdir(emotion_path): # Loop through each image in the emotion subdirectory
            if file.endswith('.jpg'):

                inputImagePath = os.path.join(emotion_path, file)
                image = cv2.imread(inputImagePath, 0)

                blurred = cv2.GaussianBlur(image, (5, 5), blur_radius)
  
                unsharp_mask = cv2.addWeighted(image, alpha, blurred, beta, 0)
                # Apply histogram equalization
                equalized_image = cv2.equalizeHist(unsharp_mask)

                # Save the sharpened image in the output subdirectory
                output_subdirectory = os.path.join(output_dir, emotion_dir, file)
                os.makedirs(os.path.join(output_dir, emotion_dir), exist_ok=True)
                cv2.imwrite(output_subdirectory, equalized_image)

print("Histogram Equalisation and Unsharp Mask Complete")
import os
import cv2
import mediapipe as mediapipe
import matplotlib.pyplot as plt

DATA_DIR = './data'

for dir in os.listdir(DATA_DIR): # Go through each directory (sign)
    dir_path = os.path.join(DATA_DIR, dir) 
    if not os.path.isdir(dir_path): # Check if path is directory
        continue

    for img_path in os.listdir(os.path.join(DATA_DIR,dir))[:1]: # Go through each image in directory
        full_img_path = os.path.join(dir_path, img_path) 
        img = cv2.imread(full_img_path)

        if img is None: # Ensure image is not empty
            print(f"Warning: Could not load image at {full_img_path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.figure()
        plt.imshow(img_rgb)

plt.show()

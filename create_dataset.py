import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence = 0.3) # Track hands only in static frame not across all (if confidence above threshold)

DATA_DIR = './data'

data = []
labels = []

for dir in os.listdir(DATA_DIR): # Go through each directory (sign)
    dir_path = os.path.join(DATA_DIR, dir) 
    if not os.path.isdir(dir_path): # Check if path is directory
        continue

    for img_path in os.listdir(os.path.join(DATA_DIR,dir))[-3:]: # Go through each image in directory
        full_img_path = os.path.join(dir_path, img_path) 
        img = cv2.imread(full_img_path)

        if img is None: # Ensure image is not empty
            print(f"Warning: Could not load image at {full_img_path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        data_sub = [] # Create an array with all XY coordinates of all landmarks of all hands in an image
        X_co = [] # To normlaise coordinates
        Y_co = []

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks: 
            for hand_landmarks in results.multi_hand_landmarks: # Go through each hand tracked/captured
                for i in range(len(hand_landmarks.landmark)): # For each landmark in image get X and Y coordinate
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    X_co.append(x)
                    Y_co.append(y)
                
                x_min = min(X_co)
                y_min = min(Y_co)

                for i in range(len(X_co)):
                    X_co[i] -= x_min
                    Y_co[i] -= y_min

                    data_sub.append(X_co[i])
                    data_sub.append(Y_co[i])
            
            data.append(data_sub)
            labels.append(dir)

# len(data) = no. of imgs, len(data[i]) = no. of landmarks
data_flattened = [np.ravel(sample) for sample in data] # Flatten each data_sub list in `data` and create a DataFrame

df = pd.DataFrame(data_flattened) # Combine data and labels into a DataFrame
df['label'] = labels
print(df)

df.to_csv('hand_landmarks.csv', index=False)
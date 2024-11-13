import os
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import pickle

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

    for img_path in os.listdir(os.path.join(DATA_DIR,dir))[:1]: # Go through each image in directory
        full_img_path = os.path.join(dir_path, img_path) 
        img = cv2.imread(full_img_path)

        if img is None: # Ensure image is not empty
            print(f"Warning: Could not load image at {full_img_path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        data_sub = [] # Create an array with all XY coordinates of all landmarks of all hands in an image
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks: 
            for hand_landmarks in results.multi_hand_landmarks: # Go through each hand tracked/captured
                for i in range(len(hand_landmarks.landmark)): # For each landmark in image get X and Y coordinate
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    data_sub.append(x)
                    data_sub.append(y)
            
            data.append(data_sub)
            labels.append(dir)

f = open('data.pickle', 'wb') # Create a file to store data
pickle.dump({'data': data, 'labels': labels}, f)
f.close()

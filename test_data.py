import numpy as np
import pandas as pd
import pickle
import cv2
import mediapipe as mp

# Load the model
with open('trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence = 0.3) # Track hands only in static frame not across all (if confidence above threshold)

while True:
    data_sub = []
    
    success, frame = cap.read()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                data_sub.append(x)
                data_sub.append(y)

        prediction = model.predict([data_sub])

        print(prediction)
            
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == 32:
        break

cap.release()
cv2.destroyAllWindows
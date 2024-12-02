import numpy as np
import pandas as pd
import pickle
import cv2
import mediapipe as mp

# Load the trained model and feature names
with open('trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

signs_dict = {'C': 'C', 'L': 'L'}

while True:
    data_sub = []
    success, frame = cap.read()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            X_co, Y_co = [], []
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                X_co.append(x)
                Y_co.append(y)
            
            # Normalize the landmarks for this hand
            x_min, y_min = min(X_co), min(Y_co)
            for i in range(len(X_co)):
                X_co[i] -= x_min
                Y_co[i] -= y_min
                data_sub.append(X_co[i])
                data_sub.append(Y_co[i])
        
        # Pad with zeros if only one hand is detected
        if len(results.multi_hand_landmarks) == 1:
            data_sub.extend([0] * 42)  # Pad 42 zeros for the second hand
        else:
            # If no hand is detected, skip this frame
            print("No hands detected. Skipping frame.")
            continue

        # Ensure the data has exactly 84 features (42 for each hand)
        if len(data_sub) < 84:
            data_sub.extend([0] * (84 - len(data_sub)))  # Pad with zeros if needed

        # Create DataFrame with the feature names loaded during training
        data_sub_df = pd.DataFrame([data_sub], columns=feature_names)

        print(data_sub_df)  # Debugging: Verify padded data

        # Make the prediction
        prediction = model.predict(data_sub_df)

        predicted_character = signs_dict[(prediction[0])]

        cv2.putText(frame, f'Prediction: {predicted_character}', (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        print(predicted_character)
            
    cv2.imshow('frame', frame)
    cv2.waitKey(25)
    if cv2.waitKey(1) & 0xFF == 32:
        break

cap.release()
cv2.destroyAllWindows()

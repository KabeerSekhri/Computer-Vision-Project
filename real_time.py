import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('sign_model.keras')

# Class labels (from train_gen or manually defined)
class_labels = ["1","5"]  # Replace with your actual class labels

# Webcam capture
cap = cv2.VideoCapture(0)

# Define constants for the box
BOX_W, BOX_H = 400, 400  # Width and height of the box

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to capture frame.")
        break

    # Flip the frame for a mirror effect
    frame = cv2.flip(frame, 1)

    # Get frame dimensions
    frame_height, frame_width = frame.shape[:2]

    # Calculate the box position (centered)
    center_x, center_y = frame_width // 2, frame_height // 2
    BOX_X, BOX_Y = center_x - (BOX_W // 2), center_y - (BOX_H // 2)

    # Draw the box on the frame
    cv2.rectangle(frame, (BOX_X, BOX_Y), (BOX_X + BOX_W, BOX_Y + BOX_H), (0, 255, 0), 2)

    # Extract the region of interest (ROI) inside the box
    roi = frame[BOX_Y:BOX_Y + BOX_H, BOX_X:BOX_X + BOX_W]

    # Preprocess the ROI for the model
    roi_resized = cv2.resize(roi, (128, 128))  # Resize to model input size
    roi_normalized = roi_resized / 255.0  # Normalize pixel values
    roi_input = np.expand_dims(roi_normalized, axis=0)  # Add batch dimension


    # Predict the class of the ROI
    predictions = model.predict(roi_input)
    print(predictions)

    class_index = np.argmax(predictions)  # Get the index of the highest prediction
    class_label = class_labels[class_index]  # Get the corresponding class label

    # Display the predicted class on the frame
    cv2.putText(frame, f'Prediction: {class_label}', (50, 50),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Real-Time Sign Detection', frame)

    # Break on 'Esc' key press
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

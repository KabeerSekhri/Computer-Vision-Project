import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained model
model = load_model('sign_model.keras')

# Image dimensions for prediction
image_x, image_y = 128, 128

# Define constants for the box
BOX_X, BOX_Y = 800, 100  # Top-left corner of the box
BOX_W, BOX_H = 400, 400  # Width and height of the box

# Pre-process the image for prediction
def keras_process_image(img):
    # Resize to (128, 128)
    img = cv2.resize(img, (image_x, image_y))
    
    # Normalize the image
    img = np.array(img, dtype=np.float32) / 255.0

    # Add batch and channel dimensions (1, 128, 128, 1)
    img = np.reshape(img, (1, image_x, image_y, 1))
    return img

# Make a prediction using the model
def keras_predict(model, image):
    processed = keras_process_image(image)
    pred_probab = model.predict(processed)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class

# Get the predicted text from the database
def get_pred_text_from_db(pred_class):
    # Dummy gesture names for example, replace with actual DB query
    gesture_names = ["C", "L"]  # Example gesture classes
    return gesture_names[pred_class]

# Main function to detect and display predictions
def main():
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        if not ret:
            break

        # Flip the frame for a mirror effect
        img = cv2.flip(img, 1)

        # Extract the region of interest (ROI) defined by the box
        roi = img[BOX_Y:BOX_Y + BOX_H, BOX_X:BOX_X + BOX_W]

        # Convert ROI to HSV and create a binary mask
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70])  # Adjust skin color range
        upper_skin = np.array([20, 255, 255])
        mask = cv2.inRange(hsv_roi, lower_skin, upper_skin)

        # Median blur for noise reduction
        mask = cv2.medianBlur(mask, 15)

        # Use the mask for prediction
        pred_probab, pred_class = keras_predict(model, mask)

        # Set a threshold for prediction confidence
        pred_text = ""
        if pred_probab * 100 > 70:  # Prediction confidence threshold
            pred_text = get_pred_text_from_db(pred_class)

        # Draw the fixed box and the prediction text on the frame
        cv2.rectangle(img, (BOX_X, BOX_Y), (BOX_X + BOX_W, BOX_Y + BOX_H), (0, 255, 0), 2)
        cv2.putText(img, f"Predicted: {pred_text}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the processed mask and the main frame
        cv2.imshow("Hand Gesture Recognition", img)
        cv2.imshow("Mask ROI", mask)

        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Press 'Esc' to quit the program
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

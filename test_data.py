import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model

# Load the pre-trained model
model = load_model('sign_model.keras')

# Get image dimensions for resizing
def get_image_size():
    img = cv2.imread('augmented_data/C/0_aug_0_rotation.jpg', 0)
    return img.shape

image_x, image_y = get_image_size()

# Pre-process the image for prediction
def keras_process_image(img):
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (1, image_x, image_y, 1))  # Add batch and channel dimensions
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
    gesture_names = ["C", "L", "A", "B"]  # Example gesture classes
    return gesture_names[pred_class]

# Get the contour and threshold for the hand sign
def get_img_contour_thresh(img):
    img = cv2.flip(img, 1)  # Flip for mirror effect
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Convert to grayscale and apply Gaussian blur for better contour detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return img, contours, thresh

# Get the prediction from the contour
def get_pred_from_contour(contour, thresh):
    x1, y1, w1, h1 = cv2.boundingRect(contour)
    save_img = thresh[y1:y1 + h1, x1:x1 + w1]  # Extract the hand region
    
    # Pre-process the extracted hand image
    pred_probab, pred_class = keras_predict(model, save_img)
    if pred_probab * 100 > 70:  # Set a threshold for prediction confidence
        text = get_pred_text_from_db(pred_class)
        return text
    return ""

# Main function to detect and display predictions
def main():
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        if not ret:
            break
        img = cv2.resize(img, (640, 480))  # Resize to standard frame size
        img, contours, thresh = get_img_contour_thresh(img)

        pred_text = ""
        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 10000:  # Only consider large contours
                pred_text = get_pred_from_contour(contour, thresh)

        # Display the predicted text
        cv2.putText(img, f"Predicted: {pred_text}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frames
        cv2.imshow("Hand Gesture Recognition", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

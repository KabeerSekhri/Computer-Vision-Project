import os
import cv2
import numpy as np

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):  # Create the 'data' directory if it doesn't exist
    os.makedirs(DATA_DIR)

# Define constants for the box
BOX_X, BOX_Y = 800, 100  # Top-left corner of the box
BOX_W, BOX_H = 400, 400  # Width and height of the box


def capture_sign(sign, dataset_size=100):
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not os.path.exists(os.path.join(DATA_DIR, str(sign))):
        os.makedirs(os.path.join(DATA_DIR, str(sign)))

    print('Collecting data for class {}'.format(sign))

    while True:  # Allow user to prepare
        success, frame = cap.read()
        if not success:
            print("Failed to access the camera.")
            break

        # Flip for mirror effect
        frame = cv2.flip(frame, 1)

        # Draw the fixed box on the screen
        cv2.rectangle(frame, (BOX_X, BOX_Y), (BOX_X + BOX_W, BOX_Y + BOX_H), (0, 255, 0), 2)
        cv2.putText(frame, 'Perform the sign inside the box', (50, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'Press SpaceBar to begin capture', (50, 400),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == 32:  # Start capture on spacebar
            break

    counter = 0
    while counter < dataset_size:
        success, frame = cap.read()
        if not success:
            print("Failed to capture frame.")
            break

        # Flip the frame for a mirror effect
        frame = cv2.flip(frame, 1)

        # Extract the region of interest (ROI) defined by the box
        roi = frame[BOX_Y:BOX_Y + BOX_H, BOX_X:BOX_X + BOX_W]

        # Preprocessing: Convert to HSV and apply skin mask
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70])  # Adjust skin color range
        upper_skin = np.array([20, 255, 255])
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        mask = cv2.medianBlur(mask, 15)

        # Find contours in the masked region
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 1000:  # Area threshold for valid signs
                # Draw contours and bounding box for feedback
                x, y, w, h = cv2.boundingRect(largest_contour)
                cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Resize the hand region and save it
                hand = mask[y:y + h, x:x + w]
                hand_resized = cv2.resize(hand, (50, 50))
                cv2.imwrite(os.path.join(DATA_DIR, str(sign), '{}.jpg'.format(counter)), hand_resized)
                counter += 1

                # Display progress
                cv2.putText(frame, f'Captured: {counter}/{dataset_size}', (50, 50),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the frame with the fixed box
        cv2.rectangle(frame, (BOX_X, BOX_Y), (BOX_X + BOX_W, BOX_Y + BOX_H), (0, 255, 0), 2)
        cv2.imshow('frame', frame)

        # Show the ROI mask in a separate window
        cv2.imshow('HSV Mask', mask)

        if cv2.waitKey(10) & 0xFF == 27:  # Press 'Esc' to quit early
            break

    cap.release()
    cv2.destroyAllWindows()


sign = ""
while not sign.strip():  # Ensure non-empty input
    sign = input("Enter the sign to capture: ").strip()
    if not sign:
        print("Input cannot be empty. Try again.")

capture_sign(sign, dataset_size=50)

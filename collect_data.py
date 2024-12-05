import os
import cv2
import numpy as np

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):  # Make a directory 'data' if not existing
    os.makedirs(DATA_DIR)

def capture_sign(sign, dataset_size=100):
    # Initialize webcam
    cap = cv2.VideoCapture(0)  
    if not os.path.exists(os.path.join(DATA_DIR, str(sign))):  
        os.makedirs(os.path.join(DATA_DIR, str(sign)))

    print('Collecting data for class {}'.format(sign))

    while True:  # Used to allow user to be prepared for capture
        success, frame = cap.read()
        cv2.putText(frame, 'Press SpaceBar to begin capture', (50, 50), 
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)  # Show capture frame
        if cv2.waitKey(1) & 0xFF == 32:  # Start capture on spacebar
            break

    counter = 0
    while counter < dataset_size:
        success, frame = cap.read()
        if not success:
            print("Failed to capture frame.")
            break
        
        # Flip the frame for mirror effect
        frame = cv2.flip(frame, 1)

        # Preprocessing
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70])  # Adjust skin color range
        upper_skin = np.array([20, 255, 255])
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        mask = cv2.medianBlur(mask, 15)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 3000:  # Adjust area threshold as needed
                # Draw bounding box around hand
                x, y, w, h = cv2.boundingRect(largest_contour)
                hand = frame[y:y+h, x:x+w]

                # Resize hand region to fixed size (e.g., 50x50)
                hand_resized = cv2.resize(hand, (50, 50))

                # Save the preprocessed hand region
                cv2.imwrite(os.path.join(DATA_DIR, str(sign), '{}.jpg'.format(counter)), hand_resized)
                counter += 1

                # Show capturing status
                cv2.putText(frame, f'Captured: {counter}/{dataset_size}', (50, 50),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Display the original frame and mask
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Draw rectangle around hand
        cv2.imshow('frame', frame)
        cv2.imshow('mask', mask)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to quit early
            break

    cap.release()
    cv2.destroyAllWindows()

sign = ""
while not sign.strip():  # Ensure non-empty input
    sign = input("Enter the sign to capture: ").strip()
    if not sign:
        print("Input cannot be empty. Try again.")

capture_sign(sign)

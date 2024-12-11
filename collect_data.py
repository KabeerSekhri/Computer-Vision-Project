import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):  # Create the 'data' directory if it doesn't exist
    os.makedirs(DATA_DIR)

# Define constants for the box
BOX_X, BOX_Y = 800, 100  # Top-left corner of the box
BOX_W, BOX_H = 400, 400  # Width and height of the box

def capture_sign(sign, dataset_size=100):
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    # Get frame dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate center of the screen
    center_x, center_y = frame_width // 2, frame_height // 2
    
    # Calculate top-left corner of the box
    BOX_X = center_x - (BOX_W // 2)
    BOX_Y = center_y - (BOX_H // 2)

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

        # Resize the ROI and save it in the original BGR format
        roi_resized = cv2.resize(roi, (50, 50))
        cv2.imwrite(os.path.join(DATA_DIR, str(sign), '{}.jpg'.format(counter)), roi_resized)
        counter += 1

        # Display progress
        cv2.putText(frame, f'Captured: {counter}/{dataset_size}', (50, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the frame with the fixed box and the ROI
        cv2.rectangle(frame, (BOX_X, BOX_Y), (BOX_X + BOX_W, BOX_Y + BOX_H), (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        cv2.imshow('ROI', roi)

        if cv2.waitKey(500) & 0xFF == 27:  # Press 'Esc' to quit early
            break

    cap.release()
    cv2.destroyAllWindows()

sign = ""
while not sign.strip():  # Ensure non-empty input
    sign = input("Enter the sign to capture: ").strip()
    if not sign:
        print("Input cannot be empty. Try again.")

capture_sign(sign, dataset_size=25)

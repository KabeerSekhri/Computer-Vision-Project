import os
import cv2

# How To Use:
# Enter the character/word for which the sign language has to be captured
# After pressing 'Spacebar' the camera captures 100 images

DATA_DIR = './data'
if not os.path.exists(DATA_DIR): # Makes a directory 'data' if not existing
    os.makedirs(DATA_DIR)


def capture_sign(sign, dataset_size = 100):
    cap = cv2.VideoCapture(0) 


    if not os.path.exists(os.path.join(DATA_DIR, str(sign))): # Make directory inside 'data' labeled j
        os.makedirs(os.path.join(DATA_DIR, str(sign)))

    print('Collecting data for class {}'.format(sign))

    while True: # Used to allow user to be prepared for capture
        ret, frame = cap.read() 
        cv2.putText(frame, 'Press SpaceBar to begin capture', (100, 50), cv2.FONT_HERSHEY_PLAIN, 1.3, (255, 255, 255), 3, cv2.LINE_AA) 
        cv2.imshow('frame', frame) # Start capture
        if cv2.waitKey(1) & 0xFF == 32:
            break

    counter = 0
    while counter < dataset_size:
        success, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(sign), '{}.jpg'.format(counter)), frame) # Store captured frame with apprp. name

        counter += 1

    cap.release()
    cv2.destroyAllWindows()

sign = ""
while not sign.strip():  # Continue looping while the input is empty
    sign = input("Enter the sign to capture: ")
    if not sign.strip():
        print("Input cannot be empty. Try again.")


capture_sign(sign)
import os
import cv2

# Right now we can define the no. of signs and no. of imgs for each sign we need to collect
# However, we need to collect all of them at once which is not optimal incase of mistakes/additions
# Need to make it so we can specify the sign we want to capture (to make data storage clearer)
# and then run it just once for that so anyone can add a sign to it, make it a function maybe.


DATA_DIR = './data'
if not os.path.exists(DATA_DIR): # Makes a directory 'data' if not existing
    os.makedirs(DATA_DIR)

number_of_classes = 1 # No. of signs to capture
dataset_size = 100 # No. of imgs to capture

cap = cv2.VideoCapture(0) 
for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))): # Make directory inside 'data' labeled j
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    done = False
    while True: # Used to allow user to be prepared for capture
        ret, frame = cap.read() 
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA) 
        cv2.imshow('frame', frame) # Start capture
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame) # Store captured frame with appr. name

        counter += 1

cap.release()
cv2.destroyAllWindows()
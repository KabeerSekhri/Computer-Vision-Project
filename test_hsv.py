import cv2
import numpy as np

cap =cv2.VideoCapture(0)
BOX_W, BOX_H = 400, 400  # Width and height of the box
while True:
    _, frame = cap.read()

    frame_h, frame_w, _ = frame.shape

    # Calculate the top-left corner of the centered ROI box
    BOX_X = ((frame_w - BOX_W) // 2)
    BOX_Y = ((frame_h - BOX_H) // 2)+50


    # Extract the region of interest (ROI) defined by the box
    roi = frame[BOX_Y:BOX_Y + BOX_H, BOX_X:BOX_X + BOX_W]
    frame = cv2.flip(frame, 1)
    cv2.rectangle(frame, (BOX_X, BOX_Y), (BOX_X + BOX_W, BOX_Y + BOX_H), (0, 255, 0), 2)
    cv2.imshow("Frame",frame)

    # Preprocessing: Convert to HSV and apply skin mask
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70])  # Adjust skin color range
    upper_skin = np.array([20, 255, 255])
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.medianBlur(mask, 15)

    cv2.imshow("mask",mask)

    #skin = cv2.bitwise_and(roi,roi, mask=mask)
    #cv2.imshow("hsv",skin)

    key= cv2.waitKey(1)
    if key == 32:
        break


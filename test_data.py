import cv2

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    
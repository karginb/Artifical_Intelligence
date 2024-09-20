import cv2
import numpy as np 

cap = cv2.VideoCapture("dog.mp4")

while True:
    ret, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sensitivity = 15
    lower_white = np.array([0, 0, 255-sensitivity])
    upper_white = np.array([255, sensitivity, 255])

    mask = cv2.inRange(hsv, lower_white, upper_white)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", res)
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break


cv2.destroyAllWindows()


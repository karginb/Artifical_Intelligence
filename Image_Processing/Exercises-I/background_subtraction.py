import cv2
import numpy as np 

cap = cv2.VideoCapture("car.mp4")
_, first_frame = cap.read()
first_frame = cv2.resize(first_frame,(640,480))


first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
first_gray = cv2.GaussianBlur(first_gray, (5,5), 0)

while 1:
    _,frame = cap.read()
    frame = cv2.resize(frame,(640,480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    diff = cv2.absdiff(first_gray, gray)
    _,diff = cv2.threshold(diff, 200, 255, cv2.THRESH_BINARY)
    cv2.imshow("Frame", frame)
    cv2.imshow("Firs_Frame", first_frame)
    cv2.imshow("diff", diff)
    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
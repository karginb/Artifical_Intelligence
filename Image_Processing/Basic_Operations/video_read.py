import cv2

cap = cv2.VideoCapture("deniz.mp4")

while True:
    ret, frame = cap.read()
    if ret == 0:
        break
    frame = cv2.flip(frame, 0)
    cv2.imshow("webcam", frame)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
    
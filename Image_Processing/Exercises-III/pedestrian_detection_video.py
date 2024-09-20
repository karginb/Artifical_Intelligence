import cv2
import imutils

cap = cv2.VideoCapture("")

while True:
    ret,frame = cap.read()
    
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    (coordinate,_) = hog.detectMultiScale(frame, winStride=(4,4), padding=(8,8), scale=1.05)
    
    for (x,y,w,h) in coordinate:
        cv2.rectangle(frame, (x,y), (x + w, y + h), (0,255,0), 3)
    
    cv2.imshow("Pedestrian", frame)
    
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

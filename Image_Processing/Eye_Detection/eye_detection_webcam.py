import cv2 

#vid = cv2.VideoCapture("eye.mp4")
vid = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("frontalface.xml")
eye_cascade = cv2.CascadeClassifier("eye.xml")



while True:
    ret, frame = vid.read()
    frame = cv2.flip(frame, 1)
    #frame = cv2.resize(frame, (640,800))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    for(x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x + w, y + h), (0,255,0), 3)
    frame2 = frame[y:y+h, x:x+w]
    gray2 = gray[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(gray2, 1.1, 5)
    
    for(ex,ey,ew,eh) in eyes:
        cv2.rectangle(frame2, (ex,ey), (ex + ew, ey + eh), (0,0,255), 3)
    
    cv2.imshow("Eyes", frame)

    if cv2.waitKey(5) & 0xFF == ord("q"):
        break


vid.release()
cv2.destroyAllWindows()

    
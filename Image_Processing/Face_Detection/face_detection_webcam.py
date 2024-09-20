import cv2

vid = cv2.VideoCapture(0)
#vid = cv2.VideoCapture("faces.mp4")

face_cascade = cv2.CascadeClassifier("frontalface.xml")

while True:
    ret,frame = vid.read()
    frame = cv2.flip(frame, 1)
    #frame = cv2.resize(frame, (540,750))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 7)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y), (x + w, y + h), (0,255,0), 2)
    cv2.imshow("Face", frame)
    
    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

vid.release()
cv2.destroyAllWindows()

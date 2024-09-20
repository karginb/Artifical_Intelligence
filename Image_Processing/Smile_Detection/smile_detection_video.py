import cv2

#vid = cv2.VideoCapture("smile.mp4")

vid = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("frontalface.xml")

smile_cascade = cv2.CascadeClassifier("smile.xml")


while True:
    ret,frame = vid.read()
    
    frame = cv2.flip(frame, 1)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x + w, y + h), (0,255,0), 3)

    roi_img = frame[y:y + h, x:x + h]
    roi_gray = gray[y:y + h, x:x + h]

    smiles = smile_cascade.detectMultiScale(roi_gray, 1.3, 3)

    for (sx,sy,sw,sh) in smiles:
        cv2.rectangle(roi_img, (sx,sy), (sx + sw, sy + sh), (0,0,255), 2)
    
    cv2.imshow("Smiles", frame)
    
    if cv2.waitKey(2) & 0xFF == ord("q"):
        break

vid.release()
cv2.destroyAllWindows()
        

import cv2

vid = cv2.VideoCapture("car.mp4")

car_cascade = cv2.CascadeClassifier("car.xml")


while True:
    ret,frame = vid.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cars = car_cascade.detectMultiScale(gray, 1.15, 7)

    for (x,y,w,h) in cars:
        cv2.rectangle(frame, (x,y), (x + w, y + h), (0,255,0), 2)
    
    cv2.imshow("Cars", frame)
    if cv2.waitKey(5) & 0xFF == ord("q"):
        break


vid.release()
cv2.destroyAllWindows()
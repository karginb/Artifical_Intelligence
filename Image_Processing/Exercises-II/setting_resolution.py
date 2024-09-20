
import cv2

cv2.namedWindow("Live Video")

cap = cv2.VideoCapture(0)

print("Width:" + str(cap.get(3)))
print("Height:" + str(cap.get(4)))

cap.set(3,1280)
cap.set(4,720)
while True:
    _,frame = cap.read()
    frame = cv2.flip(frame, 1)

    cv2.imshow("Live Video",frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
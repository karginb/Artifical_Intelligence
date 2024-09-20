import cv2
import imutils

img = cv2.imread("pedestrian.jpg")
img = imutils.resize(img, 400)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

(coordinate,_) = hog.detectMultiScale(img, winStride=(4,4), padding=(8,8), scale=1.05)


for (x,y,w,h) in coordinate:
    cv2.rectangle(img, (x,y), (x + w, y + h), (0,255,0), 3)
    cv2.imshow("Pedestrian", img)


cv2.destroyAllWindows()
cv2.waitKey(0)
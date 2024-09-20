import cv2

img = cv2.imread("body.jpg")

body_cascade = cv2.CascadeClassifier("fullbody.xml")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

bodies = body_cascade.detectMultiScale(gray, 1.8, 2)


for (x,y,w,h) in bodies:
    cv2.rectangle(img, (x,y), (x + w, y + h), (0,0,255),3)


cv2.imshow("Body",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
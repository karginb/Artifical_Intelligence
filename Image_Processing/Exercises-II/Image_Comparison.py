import numpy as np 
import cv2

img1 = cv2.imread("aircraft.jpg")
img2 = cv2.imread("aircraft.jpg")

img1 = cv2.resize(img1, (640,550))

img2 = cv2.resize(img2, (640,550))

img3 = cv2.medianBlur(img1, 7)

diff = cv2.subtract(img1, img3)
b,g,r = cv2.split(diff)

if img1.shape == img2.shape:
    print("same size")
else:
    print("not same")

if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0 :
    print("completely equal")
else:
    print("NOT completely equal")


cv2.imshow("Difference",diff)
cv2.waitKey(0)
cv2.destroyAllWindows()
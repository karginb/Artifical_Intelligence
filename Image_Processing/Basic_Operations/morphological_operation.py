import cv2
import numpy as np 
img = cv2.imread("african_elephant.jpg")
kernel = np.ones((5,5), np.uint8)
#dilation = cv2.dilate(img, kernel, iterations=5)
#opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
#opening = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
#gradient = cv2.morphologyEx(img,cv2.MORPH_GRADIENT, kernel)
#tophat = cv2.morphologyEx(img,cv2.MORPH_TOPHAT, kernel)
cv2.imshow("img",img)
cv2.imshow("tophat",tophat)
cv2.waitKey(5000)
cv2.destroyAllWindows()
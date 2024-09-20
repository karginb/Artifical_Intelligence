import cv2
import numpy as numpy
import matplotlib.pyplot as plt 

img = cv2.imread("african_elephant.jpg")
#print(img)

corner = img[0:100, 0:100]
img[0:100, 0:250] = (0, 0, 120)
cv2.imshow("Test",img)
#cv2.imshow("Corner",corner)

cv2.waitKey(1000)
cv2.destroyAllWindows()
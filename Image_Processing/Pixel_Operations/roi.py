import cv2
import numpy as np 
import matplotlib.pyplot as plt 

img = cv2.imread("african_elephant.jpg")

print(img.shape)

roi = img[200:700, 0:1000]
img[300:800, 0:1000] = roi

cv2.imshow("elephant",img)
cv2.imshow("roi",roi)
cv2.waitKey(5000)
cv2.destroyAllWindows()
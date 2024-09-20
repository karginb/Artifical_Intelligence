import cv2
import numpy as np 
from matplotlib import pyplot as plt

img = cv2.imread("african_elephant.jpg",0)
ret,th1 = cv2.threshold(img,150,200,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,21,2)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,2)

cv2.imshow("img-th1",th1)
cv2.imshow("img-th2",th2)
cv2.imshow("img-th3",th3)
cv2.waitKey(5000)
cv2.destroyAllWindows()
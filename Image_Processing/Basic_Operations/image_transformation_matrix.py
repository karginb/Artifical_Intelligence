import cv2
import numpy as np 

img = cv2.imread("african_elephant.jpg",0)
row,col = img.shape

M = np.float32([[1,0,50], [0,1,200]])

dst = cv2.warpAffine(img, M, (row,col))

cv2.imshow("dst",dst)
cv2.waitKey(5000)
cv2.destroyAllWindows
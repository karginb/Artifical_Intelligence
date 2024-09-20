import numpy as np 
import cv2
import matplotlib.pyplot as plt 

img = cv2.imread("opencv.png",0)
#print(img)
print(img.shape)
#heigh, width, channel
print("height:{} pixels".format(img.shape[0]))
print("width:{} pixels".format(img.shape[1]))
#print("chanell:{}".format(img.shape[2]))

print("Image Size:{}".format(img.size))
print("Data Type:{}".format(img.dtype))


cv2.imshow("OpenCV",img)
cv2.waitKey(1000)
cv2.destroyAllWindows()
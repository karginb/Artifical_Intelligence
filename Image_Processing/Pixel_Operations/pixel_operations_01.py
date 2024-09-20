import cv2
import numpy as np

img = cv2.imread("opencv.png")
#print(img)

(b, g, r) = img[0, 0]
print("Red:{}, Green:{}, Blue:{}".format(r,g,b))


print("Before", img[100,100])
img[100,100] = [100,100,100]
print("After:",img[100,100])


blue = img[100,100,0]
green = img[100,100,1]
red = img[100,100,2]

print("RED Value(before):", img.item(10,10,2))
img.itemset((10,10,2), 100)
print("RED Value(after):", img.item(10,10,2))

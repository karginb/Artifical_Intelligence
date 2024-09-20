import cv2
import numpy as np 
import matplotlib.pyplot as plt 

img = cv2.imread("opencv.png")
print("Shape:{}".format(img.shape))

(B, G, R) = cv2.split(img)

black_background = np.zeros(img.shape[:2], dtype = "uint8")

cv2.imshow("Blue", cv2.merge([B,black_background,black_background]))
cv2.imshow("Green", cv2.merge([black_background,G,black_background]))
cv2.imshow("Red", cv2.merge([black_background,black_background,R]))

# cv2.imshow("Opencv",img)
# cv2.imshow("Opencv-B", B)
# cv2.imshow("Opencv-G", G)
# cv2.imshow("Opencv-R", R)

cv2.waitKey(5000)
cv2.destroyAllWindows()
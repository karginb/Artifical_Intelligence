import cv2
import numpy as np

img_median = cv2.imread("median.png")
blur_m = cv2.medianBlur(img_median,5)
blur_g = cv2.GaussianBlur(img_median,(5,5), cv2.BORDER_DEFAULT)
s = cv2.bilateralFilter(img_median, 9, 95,95)

cv2.imshow("Original", img_median)
cv2.imshow("blur_m",blur_g)

cv2.waitKey(5000)
cv2.destroyAllWindows()
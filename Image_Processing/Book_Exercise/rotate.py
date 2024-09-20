import numpy as np 
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Original", image)

(h, w) = image.shape[:2]

center = (w // 2, h // 2)

M = cv2.getRotationMatrix2D(center, 45, 1.0)
print(M)
#rotated = cv2.warpAffine(image, M, (w,h))
rotated = imutils.rotate(image, 45)
cv2.imshow("Rotated by 45 degrees", rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
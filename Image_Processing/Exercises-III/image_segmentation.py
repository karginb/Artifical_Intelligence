import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("bird.jpg")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_blue = (55, 0 ,0)
upper_blue = (118, 255, 255)

mask = cv2.inRange(img_hsv, lower_blue, upper_blue)

result = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

"""cv2.imshow("Mask", mask)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()"""

plt.subplot(1,2,1)
plt.imshow(mask, cmap="gray")
plt.subplot(1,2,2)
plt.imshow(result)
plt.show()
import matplotlib.pyplot as plt
import numpy as np
import cv2

img = cv2.imread("airplane.jpg")

"""img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("img", img)

hist = cv2.calcHist([img], [0], None, [256], [0,256])
plt.figure()
plt.title("Grayscale")
plt.xlabel("Bins")
plt.ylabel("N of pixels")
plt.plot(hist)
plt.xlim([0,256])
plt.show()
cv2.waitKey(0)"""

cv2.imshow("img", img)

chans = cv2.split(img)
colors = ("b", "g", "r")
plt.figure()
plt.title("’Flattened’ Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
for (chan, color) in zip(chans, colors):
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    plt.plot(hist, color = color)
    plt.xlim([0, 256])


plt.show()

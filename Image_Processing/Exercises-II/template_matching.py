import cv2
import numpy as np 

img = cv2.imread("starwars.jpg")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

template = cv2.imread("starwars2.jpg", cv2.IMREAD_GRAYSCALE)

result = cv2.matchTemplate(gray_img, template, cv2.TM_CCOEFF_NORMED)
location = np.where(result >= 0.9)

w,h = template.shape[::-1]
for point in zip(*location[::-1]):
    cv2.rectangle(img, point, (point[0] + w, point[1] + h), (0,255,0), 3)


cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
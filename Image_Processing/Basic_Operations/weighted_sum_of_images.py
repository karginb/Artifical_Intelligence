import numpy as np
import cv2

circle = np.zeros((512, 512, 3), np.uint8) + 255
cv2.circle(circle, (256,256), 60, (255, 0, 0), -1)

rectangle = np.zeros((512, 512, 3), np.uint8) + 255
cv2.rectangle(rectangle, (150, 150), (350, 350), (0, 0, 255), -1)

dst = cv2.addWeighted(circle, 0.3, rectangle, 0.7, 0)

cv2.imshow("Circle", circle)
cv2.imshow("Rectangle", rectangle)
cv2.imshow("Dst",dst)
cv2.waitKey(5000)
cv2.destroyAllWindows()
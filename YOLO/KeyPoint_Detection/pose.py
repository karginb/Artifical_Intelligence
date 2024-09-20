import cv2
import imutils
from ultralytics import YOLO
import numpy as np 

img_path = "runs/pose/predict/powerlifter.jpg"
model_path = "yolov8n-pose.pt"

img = cv2.imread(img_path)
img = imutils.resize(img, 480)

model = YOLO(model_path)
results = model(img)[0]

for result in results:
    points = np.array(result.keypoints.xy.cpu(), dtype = "int")
    for point in points:
        for p in point:
            # print(p)
            cv2.circle(img, (p[0], p[1]), 3, (0, 255, 0), -1)

cv2.imshow("Test", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
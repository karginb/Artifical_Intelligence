import cv2
from ultralytics import YOLO
import numpy as np 
import imutils

img_path = "data/brain_tumor_dataset/images/test/620.jpg"
model_path = "runs/detect/yolov8_tumor_detection/weights/best.pt"

model = YOLO(model_path)
results = model(img_path)


img = cv2.imread(img_path)
img = imutils.resize(img, width = 360)

results = model(img)[0]

threshold = 0.5

for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result
    x1, y1, x2, y2, class_id = int(x1),int(y1), int(x2), int(y2), int(class_id)
    if score > threshold:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        class_name = results.names[class_id]
        score = score * 100
        text = f"{class_name}: &{score:.2f}"

        cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

cv2.imshow("Test Image Result",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
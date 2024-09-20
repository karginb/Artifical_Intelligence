from ultralytics import YOLO
import numpy as np
import cv2
model = YOLO("runs/classify/yolov8_covid_classification/weights/best.pt")
results = model("covid_classification/test/covid/COVID-3.png")

class_dict = results[0].names
probs = results[0].probs.data.tolist()

print(f"Classes:, {class_dict}")
print(f"Probs:, {probs}")

print(f"Results:", {class_dict[np.argmax(probs)]})

name = class_dict[np.argmax(probs)]
max_prob = int(np.max(probs) * 100)

text = name + " " + "%" + str(max_prob)

img = cv2.imread("covid_classification/test/covid/COVID-3.png")
cv2.putText(img, text, (50,50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
cv2.imshow("image",img)
import cv2
import numpy as np 
import imutils
from ultralytics import YOLO

def extract_data(img, model):
    h, w, c = img.shape
    results = model.predict(source = img.copy(), save = False, save_txt = False)
    result = results[0]
    seg_contour_idx = []

    for seg in result.masks.segments:
        seg[:,0] = seg[:,0] * w
        seg[:,1] = seg[:,1] * h
        segment = np.array(seg, dtype = np.int32)
        seg_contour_idx.append(segment)
    
    bboxes = np.array(result.boxes.xyxy.gpu(), dtype = "int")
    class_ids = np.array(result.boxes.cls.gpu(), dtype = "int")
    scores = np.array(result.boxes.conf.gpu(), dtype = "float").round(2)
    class_names = result.names

    return bboxes, class_ids, seg_contour_idx, scores, class_names

img_path = ""
model_path = ""

img = cv2.imread(img_path)
img = imutils.resize(img, width = 360)

model = YOLO(model_path)

bboxes, class_ids, seg_contour_idx, scores, class_names = extract_data(img, model)

for box, class_id, segmentation_id, score in zip(bboxes, class_ids, seg_contour_idx, scores):
  (xmin, ymin, xmax, ymax) = box

  # cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
  # cv2.polylines(img, [segmentation_id], True, (255,0,0), 2)
  cv2.fillPoly(img, [segmentation_id], color = ((0, 0, 255)))

  class_name = class_names[class_id]
  score = score * 100
  text = f"{class_name}: %{score:.2f}"
  
  cv2.putText(img, str(text), (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)


cv2.imshow(img)

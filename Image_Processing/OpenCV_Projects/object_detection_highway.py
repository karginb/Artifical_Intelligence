import cv2
import numpy as np
from tracker import *

tracker = EuclideanDistTracker()

vid = cv2.VideoCapture("highway.mp4")

object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=65)

while True:
    ret,frame = vid.read()
    
    roi = frame[340:720, 500:800]

    mask = object_detector.apply(roi)
    _,mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    detections = []


    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 400:
            #cv2.drawContours(roi, [cnt], -1, (0,255,0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x,y), (x + w, y + h), (0,255,0), 2)


            detections.append([x, y, w, h])
    
    boxes_ids = tracker.update(detections)

    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 2)
        cv2.rectangle(roi, (x,y), (x + w, y + h), (0,255,0), 3)
    

    cv2.imshow("Roi", roi)
    cv2.imshow("Mask", mask)
    cv2.imshow("Frame", frame)

    if cv2.waitKey(30) & 0xFF == ord("q"):
        break




vid.release()
cv2.destroyAllWindows()
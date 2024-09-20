import os 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import argparse
from ultralytics import YOLO
import imutils
from collections import defaultdict
import numpy as np
import time

parser = argparse.ArgumentParser(description = "Object Tracking and Counting")
parser.add_argument("-i", "--input", type = str)
parser.add_argument("-o", "--output", type = str)
parser.add_argument("-m", "--model", type = str)

args = vars(parser.parse_args())

width, height = 1280, 720
cap = cv2.VideoCapture(args["input"])
model = YOLO(args["model"])

counter = {}
track_history = defaultdict(lambda: [])
num_of_frame = 0
total_fps = 0
average_fps = 0
video_frames = []
x, y = cap.get(3), cap.get(4)

while True:
    start = time.time()
    ret, frame = cap.read()
    if ret == False:
        break

    frame = cv2.resize(frame,  (width,height))
    results = model.track(frame, persist = True, verbose = False)[0]
    bboxes = np.array(results.boxes.data.tolist(), dtype = "int")

    cv2.line(frame, (width//2, 0), (width//2, height), color = (0,0,255), thickness = 2)
    cv2.rectangle(frame, ((width//2 + 7), (height//2 - 35)), ((width //2 + 178), (height//2 - 75)), color = (255,255,255), thickness = -1)
    cv2.putText(frame, "Reference Line", ((width //2 + 10), (height//2 - 50)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color = (0,0,255), thickness = 2)

    for box in bboxes:
        x1, y1, x2, y2, track_id, score, class_id = box
        text = "ID:{} LUGGAGE".format(track_id)

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        center_coordinates = (cx, cy)
        

        if cx > width // 2 :
            cv2.circle(frame, center_coordinates, 3, (0,0,255), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
            cv2.rectangle(frame, (x1, y1), (x1 + 178, y1 - 25), (0,0,255), -1)
            cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        else:
            cv2.circle(frame, center_coordinates, 3, (0, 255,0), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255,0), 2)
            cv2.rectangle(frame, (x1, y1), (x1 + 178, y1 - 25), (0, 255,0), -1)
            cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            
            counter[track_id] = x1, y1, x2, y2
        
        number_of_luggage = len(list(counter.keys()))

        info = f"Counter: {number_of_luggage}"
        cv2.rectangle(frame, (2, 2), (150, 50), (255,255,255), -1)
        cv2.putText(frame, info, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    
    end = time.time()
    num_of_frame += 1
    fps = 1 / (end - start)
    total_fps = total_fps + fps
    average_fps = total_fps // num_of_frame

    cv2.rectangle(frame, (width, 2), (width - 140, 50), (255,0,0), -1)
    cv2.putText(frame, "FPS:" + str(np.round(fps, 3)), (width - 130, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), thickness = 2)

    video_frames.append(frame)

    cv2.imshow("Object Counting", frame)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()

print("[INFO].. Video is creating.. please wait !")
writer = cv2.VideoWriter(args["output"], cv2.VideoWriter_fourcc(*"mp4v"), average_fps, (width, height))

for frame in (video_frames):
    writer.write(frame)
writer.release()
print("[INFO].. Video is saved in " + args["output"])
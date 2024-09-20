import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np 
from shapely.geometry import Polygon
from shapely.geometry.point import Point

from ultralytics import YOLO
from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors
import time 

track_history = defaultdict(list)

current_region = None

counting_regions = [
    {
        "name": "YOLOv8 Polygon Region",
        "polygon": Polygon([(50, 80), (250, 20), (450, 80), (400, 350), (100, 350)]),  
        "counts": 0,
        "dragging": False,
        "region_color": (255, 42, 4),
        "text_color": (255, 255, 255),  
    },
    {
        "name": "YOLOv8 Rectangle Region",
        "polygon": Polygon([(200, 250), (440, 250), (440, 550), (200, 550)]),
        "counts": 0,  
        "dragging": False,
        "region_color": (37, 255, 225),  
        "text_color": (0, 0, 0),
    },
]


def mouse_callback(event, x, y, flags, param):
    global current_region

    if event == cv2.EVENT_LBUTTONDOWN:
        for region in counting_regions:
            if region["polygon"].contains(Point(x, y)):
                current_region = region
                current_region["dragging"] = True
                current_region["offset_x"] = x
                current_region["offset_y"] = y
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if current_region is not None and current_region["dragging"]:
            dx = x - current_region["offset_x"]
            dy = y - current_region["offset_y"]
            current_region["polygon"] = Polygon([(p[0] + dx, p[1] + dy) for p in current_region["polygon"].exterior.coords])
            current_region["offset_x"] = x
            current_region["offset_y"] = y
    
    elif event == cv2.EVENT_LBUTTONUP:
        if current_region is not None and current_region["dragging"]:
            current_region["dragging"] = False


def run(weights, view_img = False, save_img = False, exist_ok = False, source = None, classes = None, line_thickness = 2, track_thickness = 2, region_thickness = 2):
    video_frame_count = 0
    num_of_frame = 0 
    if not Path(source).exists():
        raise FileNotFoundError(f"Source path {source} does not exist.")
    
    model = YOLO(weights)
    model_1 = YOLO("yolov8x.pt")
    names = model_1.model.names

    cap = cv2.VideoCapture(source)
    width, height = int(cap.get(3)), int(cap.get(4))
    fps, fourcc = int(cap.get(5)), cv2.VideoWriter_fourcc(*"mp4v")

    save_dir = increment_path(Path("output") / "example", exist_ok)
    save_dir.mkdir(parents = True, exist_ok = True)
    writer = cv2.VideoWriter(str(save_dir / f"{source}"), fourcc, fps, (width, height))


    while cap.isOpened():
        start = time.time()
        ret, frame = cap.read()
        if ret == False:
            break

        video_frame_count +=1

        results = model.track(frame, persist = True, classes = classes, verbose = False)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()

            annotator = Annotator(frame, line_width = line_thickness, example = str(names))

            for box, track_id, cls in zip(boxes, track_ids, clss):
                annotator.box_label(box, str(names[cls]), color = colors(cls, True))
                bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                track = track_history[track_id]
                track.append((float(bbox_center[0]), float(bbox_center[1])))
                if len(track) > 30:
                    track.pop(0)
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed = False, color = colors(cls, True), thickness = track_thickness)

                for region in counting_regions:
                    if region["polygon"].contains(Point((bbox_center[0], bbox_center[1]))):
                        region["counts"] += 1

        for region in counting_regions:
            region_label = str(region["counts"])
            region_color = region["region_color"] 
            regiob_text_color = region["text_color"]

            polygon_coords = np.array(region["polygon"].exterior.coords, dtype = np.int32)
            centroid_x, centroid_y = int(region["polygon"].centroid.x), int(region["polygon"].centroid.y)   

            text_size, _ = cv2.getTextSize(region_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, thickness = line_thickness)

            text_x = centroid_x - text_size[0] // 2
            text_y = centroid_y - text_size[1] // 2
            cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5), region_color, -1)
            cv2.putText(frame, region_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, regiob_text_color, thickness = line_thickness)
            cv2.polylines(frame, [polygon_coords], isClosed = True, color = region_color, thickness = region_thickness)

        end = time.time()
        num_of_frame +=1
        fps = 1 / (end - start)
        total_fps = fps 
        average_fps = total_fps // num_of_frame

        cv2.putText(frame, "FPS:" + str(np.round(fps,3)), (width - 130, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), thickness = line_thickness)
 
        if view_img:
            if video_frame_count == 1:
                cv2.namedWindow("YOLOv8 Region Counter")
                cv2.setMouseCallback("YOLOv8 Region Counter", mouse_callback)
            cv2.imshow("YOLOv8 Region Counter", frame)
        
        if save_img:
            writer.write(frame)
        
        for region in counting_regions:
            region["counts"] = 0
        
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break    
        
    del video_frame_count
    writer.release()
    cap.release()
    cv2.destroyAllWindows()


    

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",type = str, default = "yolov8x.engine", help = "initial weights path")
    parser.add_argument("--source", type  = str, required = True, help = "video file path")
    parser.add_argument("--view-img", action = "store_true", help = "show results")
    parser.add_argument("--save-img", action = "store_true", help = "save results")
    parser.add_argument("--exist-ok", action = "store_true", help = "existing project")
    parser.add_argument("--classes", nargs="+", type = int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--line_thickness", type  = int, default = 2, help = "bounding box thickness")
    parser.add_argument("--track_thickness", type  = int, default = 2, help = "tracking line thickness")
    parser.add_argument("--region_thickness", type  = int, default = 2, help = "region thickness")

    return parser.parse_args()

def main(opt):
    run(**vars(opt))

if __name__ =="__main__":
    opt = parse_opt()
    main(opt)












            

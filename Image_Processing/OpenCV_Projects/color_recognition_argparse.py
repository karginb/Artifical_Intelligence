import cv2
import numpy as np 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=int, default=0, help="Indef of the video source")
args = vars(parser.parse_args())

def nothing(x):
    pass

vid = cv2.VideoCapture(args["source"])
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("Color Picker")
cv2.createTrackbar("Hue", "Color Picker", 0, 179, nothing)
cv2.createTrackbar("saturation", "Color Picker", 255, 255, nothing)
cv2.createTrackbar("Value", "Color Picker", 255, 255, nothing)

img_hsv = np.zeros((250,500,3), np.uint8)


while True:
    h = cv2.getTrackbarPos("Hue", "Color Picker")
    s = cv2.getTrackbarPos("Saturation", "Color Picker")
    v = cv2.getTrackbarPos("Value", "Color Picker")

    img_hsv[:] = (h, s, v)
    img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    frame = vid.read()[1]
    frame = cv2.flip(frame, 1)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    height, width = frame.shape[:2]

    cx = int(width / 2)
    cy = int(height / 2)
    pixel_center = hsv_frame[cy,cx]
    hue_value = pixel_center[0]

    print(pixel_center)

    color = "Undefined"

    if hue_value < 5:
        color = "RED"
    elif hue_value < 22:
        color = "ORANGE"
    elif hue_value < 33:
        color = "YELLOW"
    elif 50 < hue_value < 80:
        color = "GREEN"
    elif 95 < hue_value < 131:
        color = "BLUE"
    elif hue_value < 170:
        color = "VIOLET"
    else:
        color = "RED"

    
    pixel_center_bgr = frame[cy,cx]
    b, g, r = int(pixel_center_bgr[0]), int(pixel_center_bgr[1]), int(pixel_center_bgr[2])
    
    
    cv2.rectangle(frame, (cx - 220, 10), (cx + 200, 120), (255, 255, 255), -1)
    cv2.putText(frame, color, (cx - 200, 90), 0, 3, (b, g, r), 5)
    cv2.circle(frame, (cx,cy), 5, (25, 25, 25), 3)
    cv2.imshow("Color Picker", img_bgr)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(100) & 0xFF == ord("q"):
        break


vid.release()
cv2.destroyAllWindows()
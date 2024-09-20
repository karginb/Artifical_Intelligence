import cv2
import numpy as np
import argparse

# argument parse is used to execute python file from the terminal.
# To execute file, firstly should go to file's directory and write python3 file_name.py video.mp4
parser = argparse.ArgumentParser(description='Process a video file.')
parser.add_argument('-i', '--video', type=str, help='Path to the video file.')
parser.add_argument('-o','--output', type=str, help="Path to save the processed video.")
args = vars(parser.parse_args())



vid = cv2.VideoCapture(args["video"])

fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = vid.get(cv2.CAP_PROP_FPS)
width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(args["output"], fourcc, fps, (width,height))
video_frames = []

def nothing(x):
    pass

cv2.namedWindow("Thresholding-Trackbars")

cv2.createTrackbar("LowerH", "Thresholding-Trackbars", 0, 255, nothing)
cv2.createTrackbar("LowerS", "Thresholding-Trackbars", 0, 255, nothing)
cv2.createTrackbar("LowerV", "Thresholding-Trackbars", 0, 255, nothing)
cv2.createTrackbar("UpperH", "Thresholding-Trackbars", 0, 255, nothing)
cv2.createTrackbar("UpperS", "Thresholding-Trackbars", 0, 255, nothing)
cv2.createTrackbar("UpperV", "Thresholding-Trackbars", 0, 255, nothing)

while True:
    ret, frame = vid.read()
    video_frames.append(frame)
    if not ret:
        break
    
    #frame = cv2.resize(frame, (640,480))

    tl = (435, 585)
    bl = (25,712)
    tr = (730,585)
    br = (875,712)

    cv2.circle(frame, tl, 5, (255,0,0), -1)
    cv2.circle(frame, bl, 5, (255,0,0), -1)
    cv2.circle(frame, tr, 5, (255,0,0), -1)
    cv2.circle(frame, br, 5, (255,0,0), -1)

    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0,0], [0,480], [640,0], [640,480]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    transformed_frame = cv2.warpPerspective(frame, matrix, (640,480))


    hsv_frame = cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2HSV)
    
    lower_h = cv2.getTrackbarPos("LowerH", "Thresholding-Trackbars")
    lower_s = cv2.getTrackbarPos("LowerS", "Thresholding-Trackbars")
    lower_v = cv2.getTrackbarPos("LowerV", "Thresholding-Trackbars")
    upper_h = cv2.getTrackbarPos("UpperH", "Thresholding-Trackbars")
    upper_s = cv2.getTrackbarPos("UpperS", "Thresholding-Trackbars")
    upper_v = cv2.getTrackbarPos("UpperV", "Thresholding-Trackbars")

    lower = np.array([lower_h, lower_s, lower_v])
    upper = np.array([upper_h, upper_s, upper_v])
    mask = cv2.inRange(hsv_frame, lower, upper)

    histogram = np.sum(mask[mask.shape[0] // 2:, :], axis=0)
    midpoint = np.int8(histogram.shape[0] / 2)
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint
    

    y = 712
    
    msk = mask.copy()

    while y > 0:
        img = mask[y - 40:y, left_base - 50:left_base + 50]
        contours,_ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                left_base = left_base - 50 + cx
        

        img = mask[y-40:y, right_base - 50: right_base + 50]
        contours,_ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                right_base = right_base - 50 + cx
        
        cv2.rectangle(msk, (left_base - 50, y), (left_base + 50, y - 40), (255,255,255), 2)
        cv2.rectangle(msk, (right_base - 50, y), (right_base + 50, y - 40), (255,255,255), 2)
        y -= 40
    
    
    cv2.imshow("Original", frame)
    cv2.imshow("Birdseye", transformed_frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Lane - Detection", msk)
    #out.write(msk)
   

    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

for frame in video_frames:
    out.write(frame)


vid.release()
out.release()
cv2.destroyAllWindows()

import cv2

cap = cv2.VideoCapture("line.mp4")
circles = []

def mouse(event, x ,y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        circles.append((x,y))


cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", mouse)

while True:
    _,frame = cap.read()

    for center in circles:
        cv2.circle(frame, center, 20, (255,0,0), -1)
    cv2.imshow("Frame",frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key == ord("q"):
        circles = []


cap.release()
cv2.destroyAllWindows()
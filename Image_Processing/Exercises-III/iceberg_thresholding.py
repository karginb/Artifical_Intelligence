import cv2

img = cv2.imread("iceberg.jpg")

def nothing(x):
    pass 

cv2.namedWindow("Iceberg Trackbar")
cv2.createTrackbar("Lower", "Iceberg Trackbar", 0, 255, nothing)
cv2.createTrackbar("Upper", "Iceberg Trackbar", 0, 255, nothing)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

while True:
    Lower = cv2.getTrackbarPos("Lower", "Iceberg Trackbar")
    Upper = cv2.getTrackbarPos("Upper", "Iceberg Trackbar")

    _,threshold = cv2.threshold(gray, Lower, Upper, cv2.THRESH_BINARY)

    cv2.imshow("Image", img)
    cv2.imshow("Threshol", threshold)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cv2.destroyAllWindows()
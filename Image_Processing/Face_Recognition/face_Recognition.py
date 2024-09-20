import cv2
import face_recognition

image = cv2.imread("faces.jpg")

faceLocations = face_recognition.face_locations(image)

pt1_0 = (218,150)
pt2_0 = (373,305)

pt1_1 = (913,506)
pt2_1 = (949,542)

cv2.rectangle(image, pt1_0, pt2_0, (0,255,0), 3)
#cv2.rectangle(image, pt1_1, pt2_1, (0,255,0), 3)

cv2.imshow("Faces", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
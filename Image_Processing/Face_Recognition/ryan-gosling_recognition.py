import cv2
import face_recognition

image = cv2.imread("ryan.jpg")
image = cv2.resize(image, (480,640))

ryanImage = face_recognition.load_image_file("ryan.jpg")
ryanImageEncoding = face_recognition.face_encodings(ryanImage)[0]

test_image = cv2.imread("ryan_test.jpg")
testImage = face_recognition.load_image_file("ryan_test.jpg")
faceLoc = face_recognition.face_locations(testImage)
test_image_encoding = face_recognition.face_encodings(testImage,faceLoc)

topLeftX = 236
topLeftY = 116
bottomRightX = 390
bottomRightY = 270

matchedFace = face_recognition.compare_faces(ryanImageEncoding, test_image_encoding)


if True in matchedFace:
    cv2.rectangle(image, (topLeftX, topLeftY), (bottomRightX, bottomRightY), (0,255,0), 2)
    cv2.putText(image, "Ryan Gosling", (topLeftX, topLeftY), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))
    cv2.imshow("Face", image)
    
else:
    cv2.rectangle(image, (topLeftX, topLeftY), (bottomRightX, bottomRightY), (0,255,0), 2)
    cv2.putText(image, "Unknown", (topLeftX, topLeftY), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255))
    cv2.imshow("Face", image)


cv2.waitKey(0)
cv2.destroyAllWindows()

    
    
        
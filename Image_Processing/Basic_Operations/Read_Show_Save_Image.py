import cv2
img = cv2.imread("african_elephant.jpg")
#print(img)
cv2.imshow("image",img)
cv2.namedWindow("image",cv2.WINDOW_NORMAL)
#cv2.imwrite("BASIC_OPERATION_WITH_IMAGES/klon.jpg",img)
cv2.waitKey(5)
cv2.destroyAllWindows()
import cv2

img = cv2.imread("starwars.jpg")
blurry_img = cv2.medianBlur(img, 7)

laplacian = cv2.Laplacian(blurry_img, cv2.CV_64F).var()

print(laplacian)

if laplacian < 500:
    print("blurry image")



cv2.imshow("Image", img)
cv2.imshow("Blurry image", blurry_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
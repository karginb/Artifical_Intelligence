import cv2
def resizewithAspectRation(img, width = None, height = None, inter = cv2.INTER_AREA):
    dimension = None
    (h,w) = img.shape[:2]

    if width is None and height is None:
        return img
    
    if width is None:
        r = height / float(h)
        dimension = (int(w*r),height)

    else:
        r = width / float(w)
        dimesion = (width,int(h * r))
    

    return cv2.resize(img, dimension, interpolation=inter)


img = cv2.imread("african_elephant.jpg")
img1 = resizewithAspectRation(img, width = None, height = 600, inter = cv2.INTER_AREA)

cv2.imshow("Original", img)
cv2.imshow("Resized", img1)
cv2.waitKey(10000)
cv2.destroyAllWindows()

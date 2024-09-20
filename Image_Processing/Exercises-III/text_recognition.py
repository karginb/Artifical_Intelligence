import cv2
from PIL import Image
import pytesseract

img_to_str = Image.open("text1.png")

result = pytesseract.image_to_string(img_to_str)


with open("text_result.txt",mode="w") as file:
    file.write(result)
    print("ready!")
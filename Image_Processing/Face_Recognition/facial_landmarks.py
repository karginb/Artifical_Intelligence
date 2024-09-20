import face_recognition
from PIL import Image, ImageDraw

img = face_recognition.load_image_file("face.jpg")

landmarks = face_recognition.face_landmarks(img)

PILImage = Image.fromarray(img)
d = ImageDraw.Draw(PILImage)

for landmark in landmarks:
    for feature in landmark.keys():
        d.line(landmark[feature], width=3)


PILImage.show()
from keras.layers import Dense,Conv2D,Dropout,Flatten,Activation,MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
import matplotlib.pyplot as plt 
from glob import glob
#%%
train_path = "fruits-360/Training/"
test_path = "fruits-360/Test/"
#%%
img = load_img(train_path + "Apple Braeburn/0_100.jpg")
plt.imshow(img)
plt.axis("off")
plt.show()
#%%
x = img_to_array(img)
print(x.shape)
#%%
className = glob(train_path + "/*")
numberofClass = len(className)
print("Number of Class:",numberofClass)
#%%
model = Sequential()
model.add(Conv2D(filters = 32 , kernel_size = (3,3) , activation = "relu" , input_shape = (x.shape)))
model.add(MaxPooling2D())


model.add(Conv2D(filters = 32 , kernel_size = (3,3) , activation = "relu"))
model.add(MaxPooling2D())
         

model.add(Conv2D(filters = 64 , kernel_size = (3,3) , activation = "relu"))
model.add(MaxPooling2D())
         
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(Dropout(0.5))        
model.add(Dense(numberofClass))
model.add(Activation("softmax"))


model.compile(optimizer = "rmsprop",loss = "categorical_crossentropy",metrics=["accuracy"])

batch_size = 32 
#%%
trainGenerator = ImageDataGenerator(
    rescale= 1./255,
    zoom_range=0.5,
    horizontal_flip=True,
    vertical_flip = True,
    rotation_range=0.5
    )

testGenerator = ImageDataGenerator(rescale= 1./255)

train_generator = trainGenerator.flow_from_directory(
    train_path,
    target_size=x.shape[:2],
    batch_size=batch_size,
    color_mode = "rgb",
    class_mode = "categorical")

test_generator = testGenerator.flow_from_directory(
    test_path,
    target_size=x.shape[:2],
    batch_size=batch_size,
    color_mode = "rgb",
    class_mode = "categorical")


model.fit_generator(generator=train_generator,
                    steps_per_epoch=1600 // batch_size,
                    epochs=100,
                    validation_data=test_generator,
                    validation_steps= 800 // batch_size)




    
    

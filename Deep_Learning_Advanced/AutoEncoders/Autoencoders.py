from keras.models import Model 
from keras.layers import Input,Dense
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt 
import json,codecs
#%%

(x_train,_),(x_test,_) = fashion_mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = x_train.reshape((len(x_train)),x_train.shape[1:][0] * x_train.shape[1:][1])
x_test = x_test.reshape((len(x_test)),x_test.shape[1:][0] * x_test.shape[1:][1])

plt.imshow(x_train[1500].reshape(28,28))
plt.axis("off")
plt.show()
#%%
input_img = Input(shape = (784,)),

encoded = Dense(32,activation = "relu")(input_img)
encoded = Dense(16,activation = "relu")(encoded)
decoded = Dense(32,activation = "relu")(encoded)
decoded = Dense(32,activation = "relu")(decoded)

autoencoder = Model(input_img,decoded)

autoencoder.compile(loss = "binar",optimizer = "rmsprop")

hist = autoencoder.fit(x_train,x_train,
                       epochs = 200,
                       batch_size = 256,
                       shuffle = True,
                       validation_data = (x_train,x_train))
#%%save model 
autoencoder.save_weights("autoencoder_model.h5")
#%% evaluation 

print(hist.history.keys())

plt.plot(hist.history["loss"],label = "Train Loss")
plt.plot(hist.history["val_loss"],label = "Val Loss")
plt.legend()
plt.show()
#%%
encoder = Model(input_img,encoded)
encoded_img = encoder.predict(x_test)

plt.imshow(x_test[1500].reshape(28,28))
plt.axis("off")
plt.show()
plt.figure()
plt.imshow(encoded_img[1500].reshape(4,4))
plt.axis("off")
plt.show()
#%%
decoded_imgs = autoencoder.predict(x_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.axis("off")

    # display reconstruction
    ax = plt.subplot(2, n, i + n+1)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.axis("off")
plt.show()










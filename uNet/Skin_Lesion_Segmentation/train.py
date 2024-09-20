import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np 
import tensorflow as tf
from tensorflow import keras
import cv2
from glob import glob
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau, TensorBoard
from keras.optimizers import Adam
from keras.metrics import Recall, Precision
from model import build_unet
from metrics import dice_coef, iou

H = 256
W = 256

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    
    return x, y

def load_data(dataset_path, split = 0.1):
    images = sorted(glob(os.path.join(dataset_path, "*.jpg")))
    masks = sorted(glob(os.path.join("ISIC2018_Task1_Training_GroundTruth", "*.png")))

    test_size = int(len(images) * split)

    train_x, valid_x = train_test_split(images, test_size = test_size, random_state = 42)
    train_y, valid_y = train_test_split(masks, test_size = test_size, random_state = 42)

    train_x, test_x = train_test_split(train_x, test_size = test_size, random_state = 42)
    train_y, test_y = train_test_split(train_y, test_size = test_size, random_state = 42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def read_image(path):
    path = path.decode()
    image = cv2.imread(path, cv2.IMREAD_COLOR) # (H, W, 3)
    image = cv2.resize(image, (W,H))
    image = image / 255.0
    image = image.astype(np.float32)
    return image


def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE) #(H, W)
    x = cv2.resize(x, (W, H))
    x = x / 255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis = -1) #(H, W, 1)
    return x


def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)

        return x, y 
    
    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 3])
    y.set_shape([H, W, 1])

    return x, y

def tf_dataset(X, Y, batch):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(10)

    return dataset

if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)

    create_dir("files")

    batch_size = 4
    lr = 1e-5
    epoch = 5
    model_path = "files/model.h5"
    csv_path = "files/data.csv"

    dataset_path = "ISIC2018_Task1-2_Training_Input"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(dataset_path)

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Valid: {len(valid_x)} - {len(valid_y)}")
    print(f"Test: {len(test_x)} - {len(test_y)}")

    train_dataset = tf_dataset(train_x, train_y, batch_size)
    valid_dataset = tf_dataset(test_x, test_y, batch_size)
    
    train_steps = len(train_x) // batch_size
    valid_steps = len(test_x) // batch_size

    if len(train_x) % batch_size != 0:
        train_steps += 1
    
    if len(valid_x) % batch_size != 0:
        valid_steps += 1



    model = build_unet((H, W, 3))
    metrics = [dice_coef, iou, Recall(), Precision()]
    model.compile(
        loss = "binary_crossentropy", optimizer = Adam(lr), metrics = metrics
    )
    model.summary()

    callbacks = [
        ModelCheckpoint(model_path, verbose = 1, save_best_only = True),
        ReduceLROnPlateau(monitor = "val_loss", factor = 0.1, patience = 5, min_lr = 1e-7, verbose = 1),
        CSVLogger(csv_path),
        TensorBoard(),
        EarlyStopping(monitor = "val_loss", patience = 20, restore_best_weights = False)

    ]

    model.fit(
        train_dataset,
        epochs = epoch,
        validation_data = valid_dataset,
        steps_per_epoch = train_steps,
        validation_steps = valid_steps,
        callbacks = callbacks
    )




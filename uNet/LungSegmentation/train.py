import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np 
import cv2
from glob import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from keras.optimizers import Adam 
from keras.metrics import Recall, Precision
from model import build_unet
from metrics import dice_loss, dice_coef, iou

H = 512
W = 512

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path, split = 0.1):
    images = sorted(glob(os.path.join(path, "CXR_png", "*.png")))
    masks1 = sorted(glob(os.path.join(path, "ManualMask", "leftMask", "*.png")))
    masks2 = sorted(glob(os.path.join(path, "ManualMask", "rightMask", "*.png")))

    split_size = int(len(images) * split)

    train_x, valid_x = train_test_split(images, train_size= split_size, random_state=42)
    train_y1, valid_y1 = train_test_split(masks1, train_size= split_size, random_state=42)
    train_y2, valid_y2 = train_test_split(masks2, train_size = split_size, random_state = 42)

    train_x, test_x = train_test_split(images, train_size = split_size, random_state=42)
    train_y1, test_y1 = train_test_split(images, train_size = split_size, random_state=42)
    train_y2, test_y2 = train_test_split(images, train_size = split_size, random_state=42)

    return(train_x, train_y1, train_y2), (valid_x, valid_y1, valid_y2), (test_x, test_y1, test_y2)


def read_image(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (W, H))
    image = image / 255.0
    image = image.astype(np.float32)
    return image


def read_mask(path1, path2):
    image1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
    image = image1 + image2
    image = cv2.resize(image, (W, H))
    image = image/ np.max(image)
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis = -1)
    return image

def tf_parse(x, y1, y2):
    def _parse(x, y1, y2):
        x = x.decode()
        y1 = y1.decode()
        y2 = y2.decode()

        x = read_image(x)
        y = read_mask(y1, y2)
        return x, y
    
    x, y = tf.numpy_function(_parse, [x, y1, y2], [tf.float32, tf.float32])
    x.set_shape([H, W, 3])
    y.set_shape([H, W, 1])
    return x, y


def tf_dataset(X, Y1, Y2, batch = 8):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y1, Y2))
    dataset = dataset.shuffle(buffer_size=200)
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(4)

    return dataset



if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)

    create_dir("files")

    batch_size = 2
    lr = 1e-5
    epoch = 10
    model_path = os.path.join("files", "model.h5")
    csv_path = os.path.join("files", "data.csv")


    dataset_path = "MontgomerySet/"
    (train_x, train_y1, train_y2), (valid_x, valid_y1, valid_y2), (test_x, test_y1, test_y2) = load_data(dataset_path)

    print(f"Train: {len(train_x)} - {len(train_y1)} - {len(train_y2)}")
    print(f"Valid: {len(valid_x)} - {len(valid_y1)} - {len(valid_y2)}")
    print(f"Test: {len(test_x)} - {len(test_y1)} - {len(test_y2)}")

    train_dataset = tf_dataset(train_x, train_y1, train_y2, batch = batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y1, valid_y2, batch = batch_size)


    model = build_unet((H, W, 3))
    metrics = [dice_coef, Recall(), Precision()]
    model.compile(loss = dice_loss, optimizer = Adam(lr), metrics = metrics)

    callbacks = [
        ModelCheckpoint(model_path, verbose = 1, save_best_only = True),
        ReduceLROnPlateau(monitor = "val_loss", factor = 0.1, patience = 5, min_lr = 1e-7, verbose = 1),
        CSVLogger(csv_path)
    ]

    model.fit(train_dataset,
              epochs = epoch,
              validation_data = valid_dataset,
              callbacks = callbacks)
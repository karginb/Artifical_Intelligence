import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np 
import cv2
import pandas as pd 
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from keras.utils import CustomObjectScope
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score, jaccard_score
from metrics import dice_loss, dice_coef, iou
from train import load_data, create_dir

H = 256
W = 256

def read_image(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (W, H))
    ori_image = image
    image = image / 255.0
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis = 0) #(1, 256, 256, 3)
    return ori_image, image



def read_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (W, H))
    ori_mask = mask
    mask = mask / 255.0
    mask = mask.astype(np.int32)
    return ori_mask, mask

def save_results(ori_x, ori_y, y_pred, save_image_path):
    line = np.ones((H, 10, 3)) * 255
    
    ori_y = np.expand_dims(ori_y, axis = -1) #(256, 256, 1)
    ori_y = np.concatenate([ori_y, ori_y, ori_y], axis = -1) #(256, 256, 3)

    y_pred = np.expand_dims(y_pred, axis = -1) #(256, 256, 1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis = -1) #(256, 256, 3)

    cat_images = np.concatenate([ori_x, line, ori_y, line, y_pred * 255], axis = 1)
    cv2.imwrite(save_image_path, cat_images)


if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)

    create_dir("results")

    with CustomObjectScope({"iou": iou, "dice_coef": dice_coef}):
        model = tf.keras.models.load_model("files/model.h5")
    
    dataset_path = "ISIC2018_Task1-2_Training_Input"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(dataset_path)

    SCORE = []

    for x, y in tqdm(zip(test_x, test_y), total = len(test_x)):
        name = x.split("/")[-1]
        ori_x, x = read_image(x)
        ori_y, y = read_mask(y)

        y_pred = model.predict(x)[0] > 0.5
        print(y_pred.shape)
        y_pred = np.squeeze(y_pred, axis = -1)
        y_pred = y_pred.astype(np.int32)

        save_image_path = f"results/{name}"
        save_results(ori_x, ori_y, y_pred, save_image_path)

        y = y.flatten()
        y_pred = y_pred.flatten()

        acc_value = accuracy_score(y, y_pred)
        f1_value = f1_score(y, y_pred, labels = [0,1], average = "binary")
        jac_value = jaccard_score(y, y_pred, labels = [0,1], average = "binary" )
        recall_value = recall_score(y, y_pred, labels = [0,1], average = "binary")
        precision_value = precision_score(y, y_pred, labels = [0,1], average = "binary")
        SCORE.append([name, acc_value, f1_value, jac_value, recall_value, precision_value]) 

    
    score = [s[1:] for s in SCORE]
    score = np.mean(score, axis=0)
    print(f"Accuracy: {score[0]:0.5f}")
    print(f"F1: {score[1]:0.5f}")
    print(f"Jaccard: {score[2]:0.5f}")
    print(f"Recall: {score[3]:0.5f}")
    print(f"Precision: {score[4]:0.5f}")

    df = pd.DataFrame(SCORE, columns = ["Image Name", "Acc", "F1", "Jaccard", "Recall", "Precision"])
    df.to_csv("files/score.csv")
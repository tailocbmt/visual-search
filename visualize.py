import json
import os
from pprint import pprint
import random
import pandas as pd
import matplotlib.pyplot as plt
import cv2

PAIR_ID = 'pair_id'
STYLE = 'style'
CATEGORY_ID = 'category_id'
BOUNDING_BOX = 'bounding_box'
IMAGE_NAME = 'image_name'
X1 = 'x_1'
Y1 = 'y_1'
X2 = 'x_2'
Y2 = 'y_2'

PATH = 'deepfashion2'

TRAIN = 'train'
VAL = 'validation'

IMAGE = 'image'
ANNO = 'annos'

train_image = os.path.join(PATH, TRAIN, IMAGE)
train_label = os.path.join(PATH, TRAIN, ANNO)

val_image = os.path.join(PATH, VAL, IMAGE)
val_anno = os.path.join(PATH, VAL, ANNO)


def read_json(path):
    file = open(path, 'r')
    data = json.load(file)
    file.close()

    return data

def visualize(row):
    _, axs = plt.subplots(1, 3, figsize=(12, 12))
    axs = axs.flatten()
    img_a = cv2.imread(os.path.join(val_image, row['image_name_a']))
    img_p = cv2.imread(os.path.join(val_image, row['image_name_p']))
    img_n = cv2.imread(os.path.join(val_image, row['image_name_n']))

    images = [img_a, img_p, img_n]
    for img,ax in zip(images, axs):
        ax.imshow(img)
    plt.show()

df = pd.read_csv('deepfashion2/val_triplets.csv')

for _, row in df.iterrows():
    visualize(row)
import os
import pandas as pd
import json
from pytorch.utils.params import *

train_image = os.path.join(DATA_PATH, TRAIN_PATH, IMAGE)
train_anno = os.path.join(DATA_PATH, TRAIN_PATH, ANNOS)

val_image = os.path.join(DATA_PATH, VAL_PATH, IMAGE)
val_anno = os.path.join(DATA_PATH, VAL_PATH, ANNOS)

label_paths = os.listdir(val_anno)

def read_json(path):
    file = open(path, 'r')
    data = json.load(file)
    file.close()

    return data

def create_dataframe(anno_path: str, save_name: str):
    labels = []

    for label_path in label_paths:
        image_name = label_path.replace('.json', '.jpg')
        
        label = read_json(os.path.join(anno_path, label_path))
        pair_id = label[PAIR_ID]
        for key, item in label.items():
            if key == PAIR_ID or key == 'source':
                continue

            style = item[STYLE]
            category_id = item[CATEGORY_ID]
            x1, y1, x2, y2 = item[BOUNDING_BOX]
            landmarks = item[LANDMARKS]
            segmentation = item[SEGMENTATION]

            result = [image_name, category_id, pair_id, style, x1, y1, x2, y2, landmarks, segmentation]
            labels.append(result)

    triplet_df = pd.DataFrame(labels ,columns=[IMAGE_NAME, CATEGORY_ID, PAIR_ID, STYLE, X1, Y1, X2, Y2, LANDMARKS, SEGMENTATION])
    triplet_df.to_csv(os.path.join(DATA_PATH, save_name))

create_dataframe(train_anno, 'train.csv')
create_dataframe(val_anno, 'val.csv')
import json
import os
from pprint import pprint
import random
import pandas as pd

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

label_paths = os.listdir(val_anno)

def read_json(path):
    file = open(path, 'r')
    data = json.load(file)
    file.close()

    return data

# labels = []

# for label_path in label_paths:
#     image_name = label_path.replace('.json', '.jpg')
    
#     label = read_json(os.path.join(val_anno,label_path))
#     pair_id = label[PAIR_ID]
#     for key, item in label.items():
#         if key == PAIR_ID or key == 'source':
#             continue

#         style = item[STYLE]
#         category_id = item[CATEGORY_ID]
#         x1, y1, x2, y2 = item[BOUNDING_BOX]

#         result = [image_name, category_id, pair_id, style, x1, y1, x2, y2]
#         labels.append(result)

# triplet_df = pd.DataFrame(labels ,columns=['image_name','category_id', 'pair_id', 'style', 'x_1','y_1','x_2','y_2'])
# triplet_df.to_csv(os.path.join(PATH, 'val.csv'))

# create triplets
def create_triplet(dataframe: pd.DataFrame):

    same_category_list = [[]]
    for i in range(1, 14):
        same_category = dataframe[dataframe[CATEGORY_ID]==i].index
        same_category_list.append(same_category)

    style_zero = 0

    result_triplets = []
    groupby_pair_df = dataframe.groupby(by=['pair_id'])
    for group, data in groupby_pair_df:
        print(data)
        for row in data.itertuples():
            if getattr(row, STYLE) == 0:
                style_zero += 1
                continue
            else:
                style = getattr(row, STYLE)
                category_id = getattr(row, CATEGORY_ID)

                same_cate_style = data[(data[CATEGORY_ID] == category_id) & (data[STYLE] == style)].index
                diff_cate_style = same_category_list[category_id]
                
                for same_id in same_cate_style:
                    if same_id == getattr(row, 'Index'):
                        continue
                    same_data = dataframe.loc[same_id, ['image_name','category_id', 'style', 'x_1','y_1','x_2','y_2']]

                    sample_diff_ids = random.choices(diff_cate_style, k=5)
                    print(sample_diff_ids)
                    for diff_id in sample_diff_ids:
                        diff_data = dataframe.loc[diff_id, ['image_name','category_id', 'style', 'x_1','y_1','x_2','y_2']]
                        
                        triplet = [
                            getattr(row, IMAGE_NAME), 
                            getattr(row, CATEGORY_ID), 
                            getattr(row, STYLE),
                            getattr(row, X1),
                            getattr(row, Y1),
                            getattr(row, X2),
                            getattr(row, Y2)                            
                        ]

                        triplet.extend(same_data.values.tolist())
                        triplet.extend(diff_data.values.tolist())
                        print(len(triplet))
                        result_triplets.append(triplet)
    
    triplet_df = pd.DataFrame(result_triplets, columns=['image_name_a', 'category_id_a', 'style_a' ,'x_1_a','y_1_a','x_2_a','y_2_a', 'image_name_p','category_id_p', 'style_p', 'x_1_p','y_1_p','x_2_p','y_2_p', 'image_name_n','category_id_n', 'style_n','x_1_n','y_1_n','x_2_n','y_2_n'])

    triplet_df.to_csv('deepfashion2/train_triplets.csv', index=False)
                    
val = pd.read_csv('deepfashion2/train.csv', index_col=0)
create_triplet(val)
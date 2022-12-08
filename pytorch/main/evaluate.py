import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
######################################################################
import argparse
import json

import numpy as np
import pandas as pd
from utils.search_utils import kNN_model

#--------------------------------------------------------------------#

# Argparse 
parser = argparse.ArgumentParser()
# Path to the dataframe contains image paths, labels,...
parser.add_argument('--save_path', 
                    default='data\\model_inference\\shopping100k\\resnet50_1000\\ckpt5',
                    help='path to the save file')
# Path to the data embeddings
parser.add_argument('--emb_path', 
                    default='data\\model_inference\\shopping100k\\resnet50_1000\\ckpt5\\data_embeddings_ckpt5.npy',
                    help='Path to the embedding dataset')
# Top K nearest embedding (+1 for the query image)
parser.add_argument('--top', 
                    default=31,
                    type=int,
                    help='Top K nearest embedding (+1 for the query image)')                    
                    
def main():
    # parse the variables
    args = parser.parse_args()

    """
    1. Open file similar attributes
    2. Load the dataset embedding
    3. Load the K NN model
    4. Find whether similar attribute image is in the query or not
    5. Save the evaluate
    """
    # similar_attr = open('data\Shopping100k\shorter_sim_attr.txt', 'r')
    similar_attr = pd.read_csv("~locnt/code/lightning-hydra-template/data/shopping100k_similar.csv")

    emb_data = np.load(args.emb_path)
    print(emb_data.shape)

    nn_model = kNN_model(emb_data,args.top)

    evaluate = []
    total = 26597
    cnt = 0
    for row in similar_attr.itertuples():
        print(cnt)
        similar_list = json.loads(row.similar)
        # print(type(similar_list))
        # print(similar_list)
        if len(similar_list) == 0:
            continue
        similar_list = [int(i) for i in similar_list]
        dists, indexes = nn_model.kneighbors(emb_data[row.Index,:].reshape(1,- 1), args.top)
        arr = np.isin(np.asarray(indexes[0]), np.asarray(similar_list)).tolist()
        evaluate.append(arr)
        cnt += 1
        if cnt == total:
            break

    np.save(args.save_path + 'evaluate{}.npy'.format(args.top-1, args.emb_path), np.asarray(evaluate))
    
if __name__=="__main__":
    main()

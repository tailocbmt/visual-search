import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
######################################################################
import torch
import argparse
import pandas as pd
import numpy as np
from utils.search_utils import kNN_model
#--------------------------------------------------------------------#

# Define variables

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
# Top K nearest embedding
parser.add_argument('--top', 
                    default=71,
                    type=int,
                    help='Top K nearest embedding')                    
                    
def main():
    # parse the variables
    args = parser.parse_args()

    # Open file sim attribute
    similar_attr = open('data\Shopping100k\shorter_sim_attr.txt', 'r')
    lines = similar_attr.readlines()
    print(len(lines))
    # Test 
    test = pd.read_csv('data\Shopping100k\Attributes\shopping100k.csv')
    test['category_name'] = test['image_name'].apply(lambda x: x.split('/')[1])

    # Load the dataset embedding
    emb_data = np.load(args.emb_path)
    print(emb_data.shape)

    # KNN model
    nn_model = kNN_model(emb_data,args.top)

    evaluate = []
    for line in lines:
        similar_list = line.split()
        similar_list = [int(i) for i in similar_list]

        dists, indexes = nn_model.kneighbors(emb_data[similar_list[0],:].reshape(1,- 1), args.top)
        
        arr = np.isin(np.asarray(indexes[0]), np.asarray(similar_list)).tolist()
        print(arr)
        evaluate.append(arr)
    np.save(args.save_path + '\evaluate{}.npy'.format(args.top-1), np.asarray(evaluate))
    similar_attr.close()
if __name__=="__main__":
    main()

import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
##################################################
import torch
import argparse
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

from utils.search_utils import get_transform_multinet
from utils.data_utils import DatasetImageNet, correct_triplet
from utils.model_utils import DeepRank

BATCH_SIZE = 64
transform_val = get_transform_multinet(224)

parser = argparse.ArgumentParser()
# Path to the dataframe contains image paths, labels,...
parser.add_argument('--df_path', 
                    default='data\Consumer2Shop\siamese_data.csv',
                    help='Dataframe contains the deep fashion dataset')
# Directory to the image dir
parser.add_argument('--img_dir',
                    default='data/Consumer2Shop/',
                    help='Root dir to the image dir')
# Path to the embedding model state dict
parser.add_argument('--emb',
                    default='data\model_inference\shopping100k\Multinet\multi_net_ckpt37.pt',
                    help='Path to the embedding model state dict')    


def main():
    args = parser.parse_args()
    df = pd.read_csv(args.df_path)
    df['image_pair_name_1'] = df['image_pair_name_1'].apply(lambda x: args.img_dir + x)
    df['image_pair_name_2'] = df['image_pair_name_2'].apply(lambda x: args.img_dir + x)
    df['image_name'] = df['image_name'].apply(lambda x: args.img_dir + x)
    test = df.loc[df['evaluation_status'] =='test',:].reset_index(drop=True)

    val_dataset = DatasetImageNet(test, transform=transform_val)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = DeepRank()
    model = model.to('cuda')
    model.load_state_dict(torch.load(args.emb))
    model.eval()


    print("Generating validation data embedding...")
    triplet_ranks = 0
    batches = 0
    with torch.no_grad():
        for batch_idx, (X_val_query, X_val_positive, X_val_negative) in enumerate(valloader):

            if (X_val_query.shape[0] < BATCH_SIZE):
                continue
            
            X_val_query = Variable(X_val_query).cuda()
            X_val_positive = Variable(X_val_positive).cuda()
            X_val_negative = Variable(X_val_negative).cuda()

            batches += 1
            embedding = model(X_val_query)
            embedding_p = model(X_val_positive)
            embedding_n = model(X_val_negative)

            incorrectly_ranked_triplets = correct_triplet(embedding, embedding_p, embedding_n)
            triplet_ranks += incorrectly_ranked_triplets


    print("testing triplets ranked correctly:", (batches * BATCH_SIZE) - triplet_ranks,
        1 - float(triplet_ranks) / (batches * BATCH_SIZE))
        

if __name__ == '__main__':
    main()
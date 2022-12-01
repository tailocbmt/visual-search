import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import argparse

import numpy as np
import pandas as pd
######################################################################
import torch
from utils.data_utils import get_transform_embed
from utils.model_utils import get_embed_model
from utils.search_utils import *

#--------------------------------------------------------------------#

# Define variables
# Function including resize and nomalize
transform_embed = get_transform_embed((224, 224))

# Argparse 
parser = argparse.ArgumentParser()
# Path to the dataframe contains image paths, labels,...
parser.add_argument('--df_path', 
                    default='data\Shopping100k\Attributes\shopping100k.csv',
                    help='Dataframe contains the deep fashion dataset label')
# Path to the image dir
parser.add_argument('--img_dir',
                    default='data/Shopping100k/Images/',
                    help='Path to the embedding model state dict')                    
# Path to the data embeddings
parser.add_argument('--emb_path', 
                    default='data\model_inference\shopping100k\Multinet\multinet_resnet50_2000\ckpt4\data_embeddings.npy',
                    help='Path to the embedding dataset')
# Path to the embedding model state dict
parser.add_argument('--emb',
                    default='fashion-visual-search\\src\\pytorch\\models\\embedding_model\\multinet_VGG16bn\\resnet50_2000\\MRS4.pt',
                    help='Path to the embedding model state dict')                    
# Path to the image
parser.add_argument('--img',
                    default='data\\Shopping100k\\Images\\Female\\15_Skirts\\0071929_10.jpg',
                    help='Path to the query image')
# Number of image in the query
parser.add_argument('--top',
                    default=4,
                    help='Number of image in the query')                    



def forward(x, model, device):
    x = x.type('torch.FloatTensor').to(device)
    return(model(x))

def main():
    """
    1. read csv
    2. create labels for visualizing
    3. read the embeded data
    4. initialize kNN model
    5. create model and load model state dict
    6. read image and run through model
    7. Find the k nearest vectors
    8. Visualize
    """
    # parse the variables
    args = parser.parse_args()
    
    df = pd.read_csv(args.df_path)
    # Create the category column
    labels = create_label_shopping100k(df)    
    
    emb_data = np.load(args.emb_path)
    print(emb_data.shape)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # KNN model
    nn_model = kNN_model(emb_data,30)

    # Create the embedding model and load checkpoint
    # Turn model to evaluation mode
    emb_model = get_embed_model(128)
    emb_model = emb_model.to(device)
    state_dict = torch.load(args.emb)["state_dict"]
    
    new_state_dict = state_dict.copy()

    for old_key, value in state_dict.items():
      new_state_dict[old_key.replace("net.", "")] = value
      del new_state_dict[old_key]
    print(new_state_dict.keys())
    emb_model.load_state_dict(new_state_dict)
    emb_model.eval()


    with torch.no_grad():
        image = pil_loader(args.img)

        # Embedding Resize and convert to tensor
        im = transform_embed(image)
        im = torch.unsqueeze(im, 0)
        # Embedding
        emb = forward(im, emb_model, device).cpu().numpy()
        dist, idx = nn_model.kneighbors(emb, args.top)
    # Visualize images
    visualize(idx[0], df, labels, args.img_dir, cols=4, save=True)
    
if __name__=="__main__":
    main()

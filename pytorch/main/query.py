import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
######################################################################
import torch
import argparse
import pandas as pd
import numpy as np
from utils.search_utils import kNN_model, pil_loader, visualize, get_transform_resnet
from utils.model_utils import DeepRank, get_embed_model
#--------------------------------------------------------------------#

# Define variables
# Function including resize and nomalize
# transform_embed = get_transform_multinet(224)
transform_embed = get_transform_resnet(224)
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
                    default='data\\Shopping100k\\Images\\Female\\16_Dresses\\0000185_12.jpg',
                    help='Path to the query image')
# Number of image in the query
parser.add_argument('--top',
                    default=4,
                    help='Number of image in the query')                    

def create_label_column(df):
    df['category_name'] = df['image_name'].apply(lambda x: ' '.join(map(str, x.split('/')[1].split('_')[1:])))
    labels = df['category_name'].values.tolist()
    return labels

def forward(x, model, device):
    x = x.type('torch.FloatTensor').to(device)
    return(model(x))

def main():
    # parse the variables
    args = parser.parse_args()
    # Read_the csv
    df = pd.read_csv(args.df_path)
    # Create the category column
    labels = create_label_column(df)    
    
    emb_data = np.load(args.emb_path)
    print(emb_data.shape)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # KNN model
    nn_model = kNN_model(emb_data,30)

    # Create the embedding model and load checkpoint
    # Turn model to evaluation mode
    emb_model  = get_embed_model(2000)
    emb_model = emb_model.to('cuda')
    emb_model.load_state_dict(torch.load(args.emb))
    emb_model.eval()

    # Embedding and find the nearest vector (Euclid/ Cosine)
    with torch.no_grad():
        image = pil_loader(args.img)
        # Embedding Resize and convert to tensor
        im = transform_embed(image)
        im = torch.unsqueeze(im, 0)
        # Embedding
        emb = forward(im, emb_model, device).cpu().numpy()
        # selected_emb = selector.transform(emb)
        # print(selected_emb.shape)
        dist, idx = nn_model.kneighbors(emb, args.top)
    # Visualize images
    visualize(idx[0], df, labels, args.img_dir,cols=4)

if __name__=="__main__":
    main()

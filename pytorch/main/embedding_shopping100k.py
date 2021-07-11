import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
##################################################
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torchvision.models as models
from utils.search_utils import *
from torch.autograd import Variable
from utils.data_utils import DatasetDf
from utils.model_utils import DeepRank

# Define variables
BATCH_SIZE = 256
# Function including resize and nomalize
transform_dataset = get_transform_multinet(224)

parser = argparse.ArgumentParser()
# Path to the dataframe contains image paths, labels,...
parser.add_argument('--df_path', 
                    default='data\Shopping100k\Attributes\shopping100k.csv',
                    help='Dataframe contains the deep fashion dataset')
# Directory to the image dir
parser.add_argument('--img_dir',
                    default='data\Shopping100k\Images',
                    help='Root dir to the image dir')
# Path to the embedding model state dict
parser.add_argument('--emb',
                    default='fashion-visual-search\src\pytorch\models\embedding_model\multinet_VGG16bn\multi_net_ckpt11.pt',
                    help='Path to the embedding model state dict')    
# Output path of enbedding
parser.add_argument('--save_dir',
                    default='data\model_inference\shopping100k\Multinet\ckpt11',
                    help='Path to save file embedding')


def pil_loader(path):
    IMG = Image.open(path)

    return IMG.convert('RGB')

def main():
    # parse the variables
    args = parser.parse_args()
    # Read_the csv
    df = pd.read_csv(args.df_path)
    # Create dataset
    eval_dataset = DatasetDf(df, root_dir=args.img_dir, transform=transform_dataset, loader=pil_loader)
    evalloader = torch.utils.data.DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create the embedding model and load checkpoint
    # Turn model to evaluation mode
    emb_model = DeepRank()
    emb_model = emb_model.to(device)
    emb_model.load_state_dict(torch.load(args.emb))
    emb_model.eval()

    # Array to store embedding
    embedding = torch.randn(1,4096).type('torch.FloatTensor').to(device)
    # Embedding and find the nearest vector (Euclid/ Cosine)
    with torch.no_grad():
        for batch_idx, (eval_image, _) in enumerate(evalloader):
            # Move to cuda
            eval_image = Variable(eval_image).cuda()
            # Embedding through model
            emb = emb_model(eval_image)
            # Concat to the embedding
            embedding = torch.cat((embedding, emb), 0)
            
    # Save dataset embedding    
    embedding = np.delete(embedding.cpu().numpy(), np.s_[:1], axis=0)
    np.save(args.save_dir+'/data_embeddings',embedding)

if __name__=="__main__":
    main()
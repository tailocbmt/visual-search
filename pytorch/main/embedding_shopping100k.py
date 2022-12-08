import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import argparse

import numpy as np
import pandas as pd
##################################################
import torch
from torch.autograd import Variable
from utils.data_utils import Shopping100k
from utils.model_utils import get_embed_model
from utils.search_utils import *

# Define variables
BATCH_SIZE = 256

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

def main():
    # parse the variables
    args = parser.parse_args()
    """
    1. Read csv
    2. Load dataset
    3. Feed into DataLoader
    """
    df = pd.read_csv(args.df_path)
    eval_dataset = Shopping100k(df, im_size=(224, 224), root_dir=args.img_dir)
    evalloader = torch.utils.data.DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create the embedding model and load checkpoint
    emb_model = get_embed_model(2000)
    emb_model = emb_model.to(device)
    emb_model.load_state_dict(torch.load(args.emb))
    emb_model.eval()

    """
    1. Move batch to cuda
    2. Run through model
    3. concat to embedding matrix
    4. Save embedding matrix to file
    """
    embedding = torch.randn(1,2000).type('torch.FloatTensor').to(device)
    with torch.no_grad():
        for batch_idx, (eval_image, _) in enumerate(evalloader):            
            eval_image = Variable(eval_image).cuda()
            emb = emb_model(eval_image)
            embedding = torch.cat((embedding, emb), 0)
            
    embedding = np.delete(embedding.cpu().numpy(), np.s_[:1], axis=0)
    np.save(args.save_dir+'/data_embeddings',embedding)

if __name__=="__main__":
    main()

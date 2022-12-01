import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import argparse

import numpy as np
import pandas as pd
##################################################
import torch
from torch.autograd import Variable
from utils.data_utils import DeepFashionGallery
from utils.model_utils import get_embed_model
from utils.search_utils import *

# Define variables
BATCH_SIZE = 400

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
    df = pd.read_csv(args.df_path, skiprows=1, delimiter = "\s+")
    eval_dataset = DeepFashionGallery(df, im_size=(224, 224), root_dir=args.img_dir, source_type=1)
    
    evalloader = torch.utils.data.DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    query_dataset = DeepFashionGallery(df, im_size=(224, 224), root_dir=args.img_dir, source_type=2)
    queryloader = torch.utils.data.DataLoader(query_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create the embedding model and load checkpoint
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

    """
    1. Move batch to cuda
    2. Run through model
    3. concat to embedding matrix
    4. Save embedding matrix to file
    """
    embedding = torch.randn(1,128).type('torch.FloatTensor').to(device)
    label_list = []

    top_k_acc = 0
    mrr = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (eval_image, label) in enumerate(evalloader):
            label_list.append(label)            
            eval_image = Variable(eval_image).cuda()
            emb = emb_model(eval_image)
            embedding = torch.cat((embedding, emb), 0)
            
        embedding = np.delete(embedding.cpu().numpy(), np.s_[:1], axis=0)
        nn_model = kNN_model(embedding,30)
        label_list = torch.stack(label_list)
        print(label_list.shape)

        for batch_idx, (query_image, query_label) in enumerate(queryloader):
            query_image = Variable(query_image).cuda()
            emb = emb_model(query_image)
            dist, idx = nn_model.kneighbors(emb.cpu(), 20)

            for i in range(len(idx)):
                current_label = query_label[i]
                current_idx = idx[i, :]
                gallery_class = label_list[current_idx]
                    
                for j in range(len(gallery_class)):
                    if current_label == gallery_class[j]:
                        mrr += 1 / (j + 1)
                        top_k_acc += 1
                        break
                total += 1

    print("TOP 20 ACC: ", top_k_acc / total)
    print("MRR 20: ", mrr / total)
                

        
    # np.save(args.save_dir+'/data_embeddings',embedding)

if __name__=="__main__":
    main()
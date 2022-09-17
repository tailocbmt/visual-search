import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
##################################################
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from utils.data_utils import *
from torch.utils.data import DataLoader
from utils.model_utils import get_embed_model

parser = argparse.ArgumentParser()
# Path to the dataframe contains image paths, labels,...
parser.add_argument('--train_cfg', 
                    default='train.yml',
                    help='')
# Directory to the image dir
# parser.add_argument('--img_dir',
#                     default='fashion-visual-search',
#                     help='Root dir to the image dir')
# # Path to the model state dict
# parser.add_argument('--ckpt',
#                     default='/content/state_dict_model.pt',
#                     help='Path to the model state dict')
# # Output path of csv
# parser.add_argument('--save_dir',
#                     default='fashion-visual-search',
#                     help='Path to save file (model, loss log)')
# # Batch size              
# parser.add_argument('--batch_size',
#                     default=32,
#                     help='batch size',
#                     type=int)    
# # Number of epoch              
# parser.add_argument('--epoch',
#                     default=20,
#                     help='Number of epoch to train',
#                     type=int)
# # learning rate
# parser.add_argument('--lr',
#                     default=0.0001,
#                     help='learning rate of optimizer',
#                     type=float)

def main():
    """
    1. Read dataframe
    2. Load data (normalize, resize, ....)
    3. Split data
    4. Load into Dataloader
    """
    # Parse the agurments
    args = parser.parse_args()
    cfg = read_yml(args.train_cfg)

    if not os.path.isdir(cfg['save_dir']):
        os.makedirs(cfg['save_dir'])
    
    train_dataframe = pd.read_csv(cfg['train_df'])
    val_dataframe = pd.read_csv(cfg['val_df'])

    train_deepfashion = DeepFashion(df=train_dataframe, im_size=(224,224), root_dir=cfg['train_dir'], train=True)
    test_deepfashion = DeepFashion(df=val_dataframe, im_size=(224,224), root_dir=cfg['val_dir'], train=False)

    DATALOADER = DataLoader(train_deepfashion, batch_size=cfg['batch_size'], shuffle=True,num_workers=4)
    EVAL_DATALOADER = DataLoader(test_deepfashion, batch_size=cfg['batch_size'], shuffle=False,num_workers=4)
    print(train_deepfashion[1][0][0])
    print(train_deepfashion[1][0][0])
    
    # configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    MODEL  = get_embed_model(1000)
    MODEL = MODEL.to(DEVICE)
    MODEL_PATH = cfg['ckpt']
    # Once you have trained this model and have a checkpoint, replace None with the path to the checkpoint
    try :
        MODEL.load_state_dict(torch.load(MODEL_PATH))
    except :
        raise ValueError('Cannot load checkpoiht')

    # Loss and optimizer
    OPTIMIZER = torch.optim.SGD(MODEL.parameters(), lr = cfg['lr'], momentum=0.9)
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

    # Function forward
    def forward(x):
        x = x.type('torch.FloatTensor').to(DEVICE) 
        return(MODEL(x))

    # For updating learning rate
    def update_lr(OPTIMIZER, lr):
        for param_group in OPTIMIZER.param_groups:
            param_group['lr'] = lr

    
    print('begin training')
    LOSS_TR = []
    BIG_L = []
    TOTAL_STEP = len(DATALOADER)
    CURR_LR = cfg['lr']

    print('')
    print('')
    for epoch in range(cfg['epoch']):
        for i, (D, L) in enumerate(DATALOADER):
            print('Batch ',i,end='\r')
            #forward pass
            Q = forward(D[0])
            P = forward(D[1])
            R = forward(D[2])
            # compute loss
            loss = triplet_loss(Q,P,R)
            # Backward and optimize
            OPTIMIZER.zero_grad()
            loss.backward()
            LOSS_TR.append(loss.item())
            OPTIMIZER.step()

        print ("Epoch [{}/{}], Loss: {:.4f}".format(epoch+1, cfg['epoch'], i+1, TOTAL_STEP, np.mean(LOSS_TR)))
        BIG_L = BIG_L + np.mean(LOSS_TR)
        LOSS_TR = []

        # Evaluate on test set
        test_loss = []
        MODEL.eval()
        with torch.no_grad():
            for (D, L) in EVAL_DATALOADER:
                # Forward pass
                Q = forward(D[0])
                P = forward(D[1])
                R = forward(D[2])
                # compute loss
                loss = triplet_loss(Q,P,R)
                test_loss.append(loss.item())
            # Print loss
            print("Epoch [{}/{}], TEST Loss: {:.4f}".format(epoch+1, cfg['epoch'], np.mean(test_loss)))
            test_loss = []
        # Turn model back to train mode:
        MODEL.train()
        # Decay learning rate
        if (epoch+1) % 3 == 0:
            CURR_LR /= 1.5
            update_lr(OPTIMIZER, CURR_LR)

        # Save model checkpoint
        try :
            torch.save(MODEL.state_dict(), cfg['save_dir'] + '/MRS'+str(epoch+1)+'.pt')
            np.save(cfg['save_dir']+'/loss_file', BIG_L)
        except :
            raise ValueError('Cannot save checkpoint')

if __name__=="__main__":
    main()
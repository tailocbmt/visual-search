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
parser.add_argument('--df_path', 
                    default='fashion-visual-search\siamese_pairs_new.csv',
                    help='Dataframe contains the deep fashion dataset')
# Directory to the image dir
parser.add_argument('--img_dir',
                    default='fashion-visual-search',
                    help='Root dir to the image dir')
# Path to the model state dict
parser.add_argument('--ckpt',
                    default='/content/state_dict_model.pt',
                    help='Path to the model state dict')
# Output path of csv
parser.add_argument('--save_dir',
                    default='fashion-visual-search',
                    help='Path to save file (model, loss log)')
# Batch size              
parser.add_argument('--batch_size',
                    default=32,
                    help='batch size',
                    type=int)    
# Number of epoch              
parser.add_argument('--epoch',
                    default=20,
                    help='Number of epoch to train',
                    type=int)
# learning rate
parser.add_argument('--lr',
                    default=0.0001,
                    help='learning rate of optimizer',
                    type=float)

def main():
    # Agurment
    args = parser.parse_args()

    # Device configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('loading_data...')
    # Make dir for storing checkpoint
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    # create data loader
    dataframe = pd.read_csv(args.df_path)
    print(dataframe)
    train_deepfashion = DeepFashion(df=dataframe, root_dir=args.img_dir, train=True)
    test_deepfashion = DeepFashion(df=dataframe, root_dir=args.img_dir, train=False)

    # Split for training and testing
    torch.manual_seed(1)
    indices = torch.randperm(len(train_deepfashion)).tolist()
    test_split = 0.3
    tsize = int(len(train_deepfashion)*test_split)
    train_dataset = torch.utils.data.Subset(train_deepfashion, indices[:-tsize])
    test_dataset = torch.utils.data.Subset(test_deepfashion, indices[-tsize:])

    # Load into dataloader
    DATALOADER = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=4)
    EVAL_DATALOADER = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=4)
    print(train_deepfashion[1][0][0])
    print(train_deepfashion[1][0][0])

    #create the model
    MODEL  = get_embed_model(2000)
    MODEL = MODEL.to(DEVICE)
    MODEL_PATH = None 
    # Once you have trained this model and have a checkpoint, replace None with the path to the checkpoint
    try :
        MODEL.load_state_dict(torch.load(MODEL_PATH))
    except :
        pass

    # Loss and optimizer
    OPTIMIZER = torch.optim.SGD(MODEL.parameters(), lr = 0.0001, momentum=0.9)

    # Function forward
    def forward(x):
        x = x.type('torch.FloatTensor').to(DEVICE) 
        return(MODEL(x))

    # For updating learning rate
    def update_lr(OPTIMIZER, lr):
        for param_group in OPTIMIZER.param_groups:
            param_group['lr'] = lr

    LOSS_TR = []
    BIG_L = []

    print('begin training')
    # Train the model
    TOTAL_STEP = len(DATALOADER)
    CURR_LR = args.lr
    # define loss
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

    print('')
    print('')
    for epoch in range(args.epoch):
        for i, (D, L) in enumerate(DATALOADER):
            print('Batch ',i,end='\r')
            #forward pass
            Q = forward(D[0])
            P = forward(D[1])
            R = forward(D[2])
            # compute loss
            loss = triplet_loss(P,Q,R)
            # Backward and optimize
            OPTIMIZER.zero_grad()
            loss.backward()
            LOSS_TR.append(loss.item())
            OPTIMIZER.step()

            if (i+1) % 2000 == 0:
                temp = sum(LOSS_TR)/len(LOSS_TR)
                print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}".format(epoch+1, args.epoch, i+1, TOTAL_STEP, temp))
                BIG_L = BIG_L + LOSS_TR
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
                loss = triplet_loss(P,Q,R)
                test_loss.append(loss.item())
            # Compute loss
            print("Epoch [{}/{}], TEST SET Loss: {:.4f}".format(epoch+1, args.epoch, np.mean(test_loss)))
            test_loss = []
        # Turn to model train mode:
        MODEL.train()
        # Decay learning rate
        if (epoch+1) % 3 == 0:
            CURR_LR /= 1.5
            update_lr(OPTIMIZER, CURR_LR)

        torch.save(MODEL.state_dict(), args.save_dir + '/MRS'+str(epoch+1)+'.pt')
        try :
            np.save(args.save_dir+'/loss_file', BIG_L)
        except :
            pass

if __name__=="__main__":
    main()
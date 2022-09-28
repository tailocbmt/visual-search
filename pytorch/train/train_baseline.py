import os
from pprint import pprint
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
##################################################
import torch
import logging
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
                    
def init_logger(save_dir):
    logFormatter = logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s")
    logger = logging.getLogger()

    logger.setLevel(logging.INFO)
    fileHandler = logging.FileHandler("{0}/{1}.log".format(save_dir, 'logs'))
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)
    return logger

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
    
    logger = init_logger(cfg['save_dir'])

    train_dataframe = pd.read_csv(cfg['train_df'])
    val_dataframe = pd.read_csv(cfg['val_df'])

    train_deepfashion = DeepFashion(df=train_dataframe, im_size=(224,224), root_dir=cfg['train_dir'], train=True)
    test_deepfashion = DeepFashion(df=val_dataframe, im_size=(224,224), root_dir=cfg['val_dir'], train=False)

    DATALOADER = DataLoader(train_deepfashion, batch_size=cfg['batch_size'], shuffle=True,num_workers=4)
    EVAL_DATALOADER = DataLoader(test_deepfashion, batch_size=cfg['batch_size'], shuffle=False,num_workers=4)
    pprint(train_deepfashion[1][0][0].shape)
    pprint(train_deepfashion[1][0][0].shape)
    
    # configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    MODEL  = get_embed_model(128)
    MODEL = MODEL.to(DEVICE)
    MODEL_PATH = cfg['ckpt']
    # Once you have trained this model and have a checkpoint, replace None with the path to the checkpoint
    try :
        MODEL.load_state_dict(torch.load(MODEL_PATH))
    except:
        logger.error('Cannot load checkpoint')

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

    
    logger.info('begin training')
    train_loss = []
    losses = []
    CURR_LR = cfg['lr']

    print('')
    print('')
    for epoch in range(1, cfg['epoch']+1):
        logger.info("Epoch {epoch}:".format(epoch=epoch))
        for i, (IMAGES, L) in enumerate(DATALOADER):
            # logger.info('Current batch {batch}/{total_train_step}:'.format(batch=i, total_train_step=total_train_step))
            #forward pass
            anchor = forward(IMAGES[0])
            positive = forward(IMAGES[1])
            negative = forward(IMAGES[2])
            # compute loss
            loss = triplet_loss(anchor, positive, negative)
            # Backward and optimize
            OPTIMIZER.zero_grad()
            loss.backward()
            train_loss.append(loss.item())
            OPTIMIZER.step()

        logger.info("Epoch [{}/{}], Train Loss: {:.4f}".format(epoch, cfg['epoch'], np.mean(train_loss)))
        
        # Evaluate on test set
        test_loss = []
        MODEL.eval()
        with torch.no_grad():
            for (IMAGES, L) in EVAL_DATALOADER:
                # Forward pass
                anchor = forward(IMAGES[0])
                positive = forward(IMAGES[1])
                negative = forward(IMAGES[2])
                # compute loss
                loss = triplet_loss(anchor, positive, negative)
                test_loss.append(loss.item())
            # Print loss
            logger.info("Epoch [{}/{}], Test Loss: {:.4f}".format(epoch+1, cfg['epoch'], np.mean(test_loss)))
            losses.append([epoch, np.mean(train_loss), np.mean(test_loss)])
        pprint(losses)
        train_loss = []
        test_loss = []

        # Turn model back to train mode:
        MODEL.train()
        # Decay learning rate
        if (epoch+1) % 3 == 0:
            CURR_LR /= 1.5
            update_lr(OPTIMIZER, CURR_LR)

        # Save model checkpoint
        try :
            torch.save(MODEL.state_dict(), os.path.join(cfg['save_dir'], 'Epoch_'+str(epoch+1)+'.pt'))
            dataframe = pd.DataFrame(losses, columns=["epoch", "train_loss", "test_loss"])
            dataframe.to_csv(os.path.join(cfg['save_dir'], "train_log.csv"), index=False)
        except :
            logger.error(f"Cannot save checkpoint at {cfg['save_dir']}/{'Epoch_'+str(epoch+1)+'.pt'}")
            raise ValueError('Cannot save checkpoint')
            

if __name__=="__main__":
    main()
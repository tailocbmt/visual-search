import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
##################################################
import argparse

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms

from utils.data_utils import DatasetImageNet, correct_triplet
from utils.model_utils import DeepRank, get_embed_model

BATCH_SIZE = 512
LEARNING_RATE = 0.001
VERBOSE = 1

if VERBOSE:
    print("Libs locked and loaded")

use_cuda = torch.cuda.is_available()
print("Cuda?: " + str(use_cuda))

transform_train = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
df = pd.read_csv('/content/drive/MyDrive/modelCode/Visual-search/data/siamese_data.csv')
df['image_pair_name_1'] = df['image_pair_name_1'].apply(lambda x: os.path.join('/content', x))
df['image_pair_name_2'] = df['image_pair_name_2'].apply(lambda x: os.path.join('/content', x))
df['image_name'] = df['image_name'].apply(lambda x: os.path.join('/content', x))

train_dataset = DatasetImageNet(df, transform=transform_train)
val_dataset = DatasetImageNet(df, transform=transform_test)

# Split for training and testing
torch.manual_seed(1)
indices = torch.randperm(len(train_dataset)).tolist()
test_split = 0.3
tsize = int(len(train_dataset)*test_split)
train_dataset = torch.utils.data.Subset(train_dataset, indices[:-tsize])
test_dataset = torch.utils.data.Subset(val_dataset, indices[-tsize:])

# Load into dataloader
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
valloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


def train_and_eval_model(num_epochs, optim_name, resume, save_dir):
    conv_net = get_embed_model(2000)
    conv_net = conv_net.cuda()
    conv_net.load_state_dict(torch.load('/content/drive/MyDrive/resnet_2000/MRS5.pt'))

    model = DeepRank(conv_net)

    sub_model = list(model.conv_model.children())
    for s in sub_model:
      for param in s.parameters():
          param.requires_grad = False  # switch off all gradients except last two
    for name, param in model.conv_model.named_parameters():
      print(name)

    if use_cuda:
        model.cuda()

    # Load weight
    if resume:
        try: 
            model.load_state_dict(torch.load(resume))
        except:
            raise ValueError('Not found checkpoint path')

    if optim_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    elif optim_name == "rms":
        optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
    else:
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9,
                              nesterov=True)

    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

    model.train()
    history = []
    for epoch in range(0, num_epochs):
        if VERBOSE:
            print("Epoch is " + str(epoch+1))

        batches = 0
        train_triplet_ranks = 0
        losses_acc = []
        train_loss = []
        for batch_idx, (X_train_query, X_train_postive, X_train_negative) in enumerate(trainloader):

            if (X_train_query.shape[0] < BATCH_SIZE):
                continue

            if use_cuda:
                X_train_query = Variable(X_train_query).cuda()
                X_train_postive = Variable(X_train_postive).cuda()
                X_train_negative = Variable(X_train_negative).cuda()
            else:
                X_train_query = Variable(X_train_query)
                X_train_postive = Variable(X_train_postive)
                X_train_negative = Variable(X_train_negative)

            optimizer.zero_grad()  # set gradient to 0

            query_embedding = model(X_train_query)
            positive_embedding = model(X_train_postive)
            negative_embedding = model(X_train_negative)

            loss = triplet_loss(anchor=query_embedding, positive=positive_embedding,
                                         negative=negative_embedding)
            loss.backward()
            train_loss.append(loss.item())
            optimizer.step()

            # Calculate accuracy
            batches += 1     
            incorrectly_ranked_triplets = correct_triplet(query_embedding, positive_embedding, negative_embedding)
            train_triplet_ranks += incorrectly_ranked_triplets

        print("train triplets ranked correctly:", (batches * BATCH_SIZE) - train_triplet_ranks,
            1 - float(train_triplet_ranks) / (batches * BATCH_SIZE))

        # Print Loss:
        loss_epoch = np.mean(train_loss)
        print("Loss at epoch [{}/{}]: {}".format(epoch+1, num_epochs, loss_epoch))
        losses_acc.append(loss_epoch)
        losses_acc.append(1 - float(train_triplet_ranks) / (batches * BATCH_SIZE))

        # Save model state dict
        torch.save(model.state_dict(),save_dir + 'multi_net_ckpt'  + str(epoch + 5) + '.pt')  # temporary model to save
    
        # Evaluate on test set
        model.eval()
        val_loss = []
        val_triplet_ranks = 0
        batches = 0
        with torch.no_grad():
            for batch_idx, (X_val_query, X_val_postive, X_val_negative) in enumerate(valloader):
                # Forward pass
                if use_cuda:
                    X_val_query = Variable(X_val_query).cuda()
                    X_val_postive = Variable(X_val_postive).cuda()
                    X_val_negative = Variable(X_val_negative).cuda()
                else:
                    X_val_query = Variable(X_val_query)
                    X_val_postive = Variable(X_val_postive)
                    X_val_negative = Variable(X_val_negative)
                # compute loss
                query_embedding = model(X_val_query)
                positive_embedding = model(X_val_postive)
                negative_embedding = model(X_val_negative)

                loss = triplet_loss(anchor=query_embedding, positive=positive_embedding,
                                         negative=negative_embedding)
                val_loss.append(loss.item())
                # Calculate accuracy
                batches += 1     
                incorrectly_ranked_triplets = correct_triplet(query_embedding, positive_embedding, negative_embedding)
                val_triplet_ranks += incorrectly_ranked_triplets


            print("test triplets ranked correctly:", (batches * BATCH_SIZE) - val_triplet_ranks,
                1 - float(val_triplet_ranks) / (batches * BATCH_SIZE))

            # Compute loss
            val_loss_epoch = np.mean(val_loss)
            print("Val loss at epoch [{}/{}]: {}".format(epoch+1, num_epochs, val_loss_epoch))
        # Turn to model train mode:
        # Append loss and accuracy:
        losses_acc.append(val_loss_epoch)
        losses_acc.append(1 - float(val_triplet_ranks) / (batches * BATCH_SIZE))

        history.append(losses_acc)
        model.train()


    np.asarray(history).astype('float32').tofile('/content/mean_training_loss.txt')

    torch.save(model, save_dir+'/deepranknet.model')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optional app description')

    parser.add_argument('--epochs',
                        help='A argument for no. of epochs')

    parser.add_argument('--optim',
                        help='A argument for the optimizer to choose eg: adam')

    parser.add_argument('--resume',
                        help='A argument for checkpoint',
                        default=None)

    parser.add_argument('--save_dir',
                        help='A argument for saving checkpoint path',
                        default='/content/drive/MyDrive/modelCode/Visual-search/embedding/Multi-net/')

    args = parser.parse_args()
    epochs = 0
    if int(args.epochs) < 0:
        print('This should be a positive value')
        quit()
    else:
        epochs = int(args.epochs)

    if str(args.optim) not in ["adam", "rms"]:
        print('switching to default optimizer, SGD+Momentum')
        optim_name = "sgd"
    else:
        optim_name = args.optim.lower()

    resume = args.resume
    save_dir = args.save_dir
    train_and_eval_model(epochs, optim_name, resume, save_dir)
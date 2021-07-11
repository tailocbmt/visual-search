import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
##################################################
import torch
import cv2
import torchvision
from PIL import Image
import numpy as np
import albumentations as A
import torch.nn.functional as F
from albumentations.pytorch.transforms import ToTensorV2

def pil_loader(element):
    path = element[0]
    x1,y1,x2,y2 = element[1],element[2],element[3],element[4]
    IMG = Image.open(path)
    crop = IMG.crop((x1,y1,x2,y2))
    
    return crop.convert('RGB')
class ImagesDataset(torch.utils.data.Dataset):
    def __init__(self, files_dir, dataframe, width, height, transforms=None):
        self.transforms = transforms
        self.dataframe = dataframe
        self.files_dir = files_dir
        self.height = height
        self.width = width

        # sorting the images for consistency
        # To get images, the extension of the filename is checked to be jpg
        self.imgs = self.dataframe['image_name'].values.tolist()

        self.label_list = self.dataframe['category_type'].values.tolist() 
        # classes: 0 index is reserved for background
        #self.classes = [_, 'top', 'bottom']

    def __getitem__(self, idx):

        img_name = self.imgs[idx]
        image_path = os.path.join(self.files_dir, img_name)

        # reading the images and converting them to correct size and color    
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_res = cv2.resize(img_rgb, (self.width, self.height), cv2.INTER_AREA)
        # diving by 255
        img_res /= 255.0
        
        # annotation file
               
        boxes = []
        labels = []
        h,w,_ = img_rgb.shape

        # Convert normal coordinates bbox to resized image bbox
        box = self.dataframe.loc[idx, ['x_1', 'y_1', 'x_2', 'y_2']].values.tolist()
        
        box[0], box[1], box[2], box[3] = (box[0]/w)*self.width, (box[1]/h)*self.height, (box[2]/w)*self.width, (box[3]/h)*self.height
        boxes.append(box)
        labels.append(self.label_list[idx])

        # convert boxes into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # getting the areas of the boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        #Convert the label to tensor
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        # image_id
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        if self.transforms:
            sample = self.transforms(image = img_res,
                                     bboxes = target['boxes'],
                                     labels = labels)
            
            img_res = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])
            
        return img_res, target

    def __len__(self):
        return len(self.imgs)

# Send train=True fro training transforms and False for val/test transforms
def get_transform(train):
    
    if train:
        return A.Compose([
                     # ToTensorV2 converts image to pytorch tensor without div by 255
                            ToTensorV2(p=1.0) 
                        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
    else:
        return A.Compose([
                            ToTensorV2(p=1.0)
                        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})



class DeepFashion(torch.utils.data.Dataset):
    """ Custom dataset with triplet sampling, for the tiny image net dataset"""

    def __init__(self, df, root_dir, train, transform=None, loader = pil_loader):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if transform == None :
            if train == True:
                transform = torchvision.transforms.Compose([
                                    torchvision.transforms.Resize((224,224)),
                                    torchvision.transforms.RandomHorizontalFlip(p=0.5),
                                    torchvision.transforms.ToTensor()])
            else:
                transform = torchvision.transforms.Compose([
                                    torchvision.transforms.Resize((224,224)),
                                    torchvision.transforms.ToTensor()])
        self.df = df.copy()
        self.root_dir = root_dir
        self.transform = transform
        self.loader = loader
            
        self.df['image_pair_name_1'] = self.df['image_pair_name_1'].apply(lambda x: os.path.join(self.root_dir, x))
        self.df['image_pair_name_2'] = self.df['image_pair_name_2'].apply(lambda x: os.path.join(self.root_dir, x))
        self.df['image_name'] = self.df['image_name'].apply(lambda x: os.path.join(self.root_dir, x))
        
    def _sample(self,idx):
        p1 = self.df.loc[idx, ['image_pair_name_1','x_1_a','y_1_a','x_2_a','y_2_a']].values.tolist()
        p2 = self.df.loc[idx, ['image_pair_name_2','x_1_p','y_1_p','x_2_p','y_2_p']].values.tolist()
        p3 = self.df.loc[idx, ['image_name','x_1','y_1','x_2','y_2']].values.tolist()

        return [p1, p2, p3]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        paths = self._sample(idx)
        images = []
        for i in paths:
            temp = self.loader(i)
            temp = self.transform(temp)
            images.append(temp)
        return (images[0],images[1],images[2]),0

class DatasetDf(torch.utils.data.Dataset):
    """ Custom dataset with triplet sampling, for the tiny image net dataset"""

    def __init__(self, df, root_dir, transform=None, loader = pil_loader):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if transform == None :
            transform = torchvision.transforms.Compose([
                                torchvision.transforms.Resize((300,300)),
                                torchvision.transforms.ToTensor()])
        self.df = df.copy()
        self.root_dir = root_dir
        self.transform = transform
        self.loader = loader
            
        self.df['image_name'] = self.df['image_name'].apply(lambda x: os.path.join(self.root_dir, x))
        
    def _sample(self,idx):
        p3 = self.df.loc[idx, 'image_name']
        return p3

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self._sample(idx)
        
        temp = self.loader(path)
        temp = self.transform(temp)
        return temp,0

#############################################################################################################
# "Deep rank dataloader"
class DatasetImageNet(torch.utils.data.Dataset):
    """
    Dataset class for tiny Image net dataset
    """

    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        row = self.data.iloc[index, :].values.tolist()
        images = [pil_loader(row[0:5]), pil_loader(row[5:10]), pil_loader(row[10:15])]  # open triplet images as RGB

        if self.transform is not None:
            for i in range(0, 3):
                images[i] = self.transform(images[i])

        q_image, p_image, n_image = images[0], images[1], images[2]

        return q_image, p_image, n_image

# Calculate accuracy
def correct_triplet(anchor, positive, negative, size_average=False):
    """calculate triplet hinge loss
    Parameters
    ----------
    anchor
    positive
    negative
    size_average
    Returns
    -------
    float, denotes number of incorrect triplets
    """
    distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
    distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
    losses = F.relu(distance_positive - distance_negative + 1.0)
    losses = (losses > 0)
    # print(losses)
    return losses.mean() if size_average else losses.sum()

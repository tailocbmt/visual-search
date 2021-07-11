import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
##################################################
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as f
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchsummary import summary

def get_embed_model(num_classes, name='resnet'):

    model = None
    if name == 'resnet':
        model =  torchvision.models.resnet50(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features,num_classes)
    elif name == 'vgg16':
        model =  ConvNet_VGG16bn(num_classes)
    return model

def get_fasterCNN_model(num_classes):

    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 

    return model

class ConvNet_Resnet(nn.Module):
    """EmbeddingNet using ResNet-101."""

    def __init__(self):
        """Initialize EmbeddingNet model."""
        super(ConvNet_Resnet, self).__init__()

        # Everything except the last linear layer
        resnet = torchvision.models.resnet101(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        num_ftrs = resnet.fc.in_features
        self.fc1 = nn.Linear(num_ftrs, 4096)

    def forward(self, x):
        """Forward pass of EmbeddingNet."""

        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)

        return out

class ConvNet_VGG16bn(nn.Module):
    """EmbeddingNet using VGG16 Batch Norm."""

    def __init__(self, num_classes=4096):
        """Initialize EmbeddingNet model."""
        super(ConvNet_VGG16bn, self).__init__()

        # Everything except the last linear layer
        self.vgg16_bn = torchvision.models.vgg16_bn(pretrained=True)
        self.vgg16_bn.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.6),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.6),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        """Forward pass of EmbeddingNet."""

        out = self.vgg16_bn(x)
        return out

class DeepRank(nn.Module):
    """
    Deep Image Rank Architecture
    """

    def __init__(self, conv_model=None):
        super(DeepRank, self).__init__()
        if conv_model==None:
            self.conv_model = ConvNet_VGG16bn()  # ResNet101
        else:
            self.conv_model = conv_model

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=8, padding=1,
                                     stride=16)  # 1st sub sampling
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=4, padding=1)

        self.conv2 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=8, padding=4,
                                     stride=32)  # 2nd sub sampling
        self.maxpool2 = nn.MaxPool2d(kernel_size=7, stride=2, padding=3)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.dense_layer = nn.Linear(in_features=(2000 + 192), out_features=1000)

    def forward(self, X):
        conv_input = self.conv_model(X)
        conv_norm = conv_input.norm(p=2, dim=1, keepdim=True)
        conv_input = conv_input.div(conv_norm.expand_as(conv_input))
        
        first_input = self.conv1(X)
        first_input = self.maxpool1(first_input)
        first_input = self.avgpool(first_input)
        first_input = torch.flatten(first_input, 1)
        first_norm = first_input.norm(p=2, dim=1, keepdim=True)
        first_input = first_input.div(first_norm.expand_as(first_input))

        second_input = self.conv2(X)
        second_input = self.maxpool2(second_input)
        second_input = self.avgpool(second_input)
        second_input = torch.flatten(second_input, 1)
        second_norm = second_input.norm(p=2, dim=1, keepdim=True)
        second_input = second_input.div(second_norm.expand_as(second_input))

        merge_subsample = torch.cat([first_input, second_input], 1)  # batch x (192)

        merge_conv = torch.cat([merge_subsample, conv_input], 1)  # batch x (4096 + 3072)

        final_input = self.dense_layer(merge_conv)
        final_norm = final_input.norm(p=2, dim=1, keepdim=True)
        final_input = final_input.div(final_norm.expand_as(final_input))

        return final_input

if __name__=="__main__":
    test = DeepRank()
    test = test.cuda()
    summary(test, (3, 224, 224))
import os
import math
import sys
import argparse

import numpy as np
import pandas as pd
import cv2
import transforms as T
import PIL
from PIL import Image
from skimage import transform as sktsf

import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms as tvtsf
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from engine import train_one_epoch, evaluate
import utils


parser = argparse.ArgumentParser()

parser.add_argument("-p", "--path", dest="train_path", help="Path to training data images.", default="data/train")
parser.add_argument("-l", "--label", dest="train_label", help="Path to training data labels.", default="data/train.csv")
parser.add_argument("--hf", dest="horizontal_flips", help="Augment with horizontal flips in training. (Default=false).", action="store_true", default=False)
parser.add_argument("-e","--num_epochs", type=int, dest="num_epochs", help="Number of epochs.", default=10)
parser.add_argument("--cf","--check_freq", type=int, dest="check_freq", help="Checkpoint frequency.")
parser.add_argument("--output_weight_path", dest="output_weight_path", help="Output path for weights.", default='saved_model')
parser.add_argument("--input_weight_path", dest="input_weight_path", help="Input path for weights.")

options = parser.parse_args()

class TableDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files
        self.imgs = list(sorted(os.listdir(os.path.join(root, options.train_path))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, options.train_path, self.imgs[idx])
        # image = Image.open(img_path).convert("RGB")
        image = utils.read_image(img_path)

        # Rescaling Images
        C, H, W = image.shape
        min_size = 600
        max_size = 1024
        scale1 = min_size / min(H, W)
        scale2 = max_size / max(H, W)
        scale = min(scale1, scale2)
        image = image / 255.
        image = sktsf.resize(image, (C, H * scale, W * scale), mode='reflect',anti_aliasing=False)

        # Normalizing image
        normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        image = normalize(torch.from_numpy(image))
        image = image.numpy()
        
        if not os.path.isfile(os.path.join(self.root, options.train_label)):
            print('[error]', options.train_label+' file not found')
            exit(0)

        train_labels = pd.read_csv(os.path.join(self.root, options.train_label))

        old_boxes = []
        num_objs = 0
        for i in range(train_labels['image_id'].count()):
            if (self.imgs[idx] == train_labels['image_id'][i]):
                xmin = train_labels['xmin'][i]
                ymin = train_labels['ymin'][i]
                xmax = train_labels['xmax'][i]
                ymax = train_labels['ymax'][i]
                num_objs += 1
                old_boxes.append([xmin, ymin, xmax, ymax])
    
        # Rescale bounding box
        _, o_H, o_W = image.shape
        scale = o_H / H
        bbox = np.stack(old_boxes).astype(np.float32)
        resized_boxes = utils.resize_bbox(bbox, (H, W), (o_H, o_W))
        
        # resized boxes are stacked (R, 4) 
        # where R is the number of bboxes in the image 
        # converted it back to simple 2d-array [[xmin1, ymin1, xmax1, ymax1], ...]
        boxes = []
        for i in resized_boxes:
            box = []
            [box.append(int(b)) for b in i]
            boxes.append(box)

        # converting arrays into torch tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        image = image.transpose((1,2,0))

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.imgs)

def get_model_resnet(num_classes):
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
    return model

def get_transform(train):
    transforms = []

    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        if options.horizontal_flips:
            # during training, randomly flip the training images
            # and ground-truth for data augmentation
            transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

dataset = TableDataset(os.getcwd(), get_transform(train=True))
dataset_test = TableDataset(os.getcwd(), get_transform(train=False))

# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-1])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-1:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=True, num_workers=2,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=2,
    collate_fn=utils.collate_fn)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and table
num_classes = 2

# get the model using our helper function
model = get_model_resnet(num_classes)

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

# optimizer = torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

# create the summary writer
# writer = SummaryWriter()
# let's train it for 10 epochs
num_epochs = options.num_epochs
# step = 0

print('[info] total epochs:', num_epochs)
if options.check_freq:
    print('[info] model weights will saved in /'+options.output_weight_path+' folder after every', options.check_freq, 'epochs\n')
else:
    print('[info] model weights will saved in /'+options.output_weight_path+' folder\n')

for epoch in range(num_epochs):
    # train for one epoch, printing every 100 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=100)
    ## you can paste train_one_epoch function code here ##
    #     step += 1

    #     # write scalars to tensorboard after 100 iterations
    #     if (step % 100 == 0):
    #         for key, val in loss_dict.items():
    #             # adding 4 losses one by one
    #             writer.add_scalar(key, val.item(), step)
    #         # adding total loss
    #         writer.add_scalar('loss: ', loss_value, step)

    # update the learning rate
    # if lr_scheduler is not None:
    lr_scheduler.step()

    # print('evaluating...')
    # evaluate on the test dataset
    # evaluate(model, data_loader_test, device=device)

    # check if output_weight_path directory exists
    if not os.path.exists(options.output_weight_path):
        os.makedirs(options.output_weight_path)

    # if model checkpoints frequency is provided
    if options.check_freq:
        # save model weights at given frequency
        if ((epoch+1)%options.check_freq == 0):
            torch.save(model.state_dict(), options.output_weight_path+'/model_ep-{}.pth'.format(epoch+1))
    # save the last weights 
    if ((epoch+1) == num_epochs):
        torch.save(model.state_dict(), options.output_weight_path+'/model_ep-{}.pth'.format(epoch+1))

    torch.cuda.empty_cache()

# writer.close()
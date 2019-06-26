import os
import math
import sys

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


class TableDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files
        self.imgs = list(sorted(os.listdir(os.path.join(root, "train"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "train", self.imgs[idx])
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
    
        train_labels = pd.read_csv('data/train.csv')

        ######## REMOVE THIS ###########
        # old_img = cv2.imread(img_path)
        ################################

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

                ##################### REMOVE THIS #############################
                # cv2.rectangle(old_img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
                ###############################################################
            
        # Rescale bounding box
        _, o_H, o_W = image.shape
        scale = o_H / H
        bbox = np.stack(old_boxes).astype(np.float32)
        resized_boxes = utils.resize_bbox(bbox, (H, W), (o_H, o_W))
        
        boxes = []
        for i in resized_boxes:
            box = []
            [box.append(int(b)) for b in i]
            boxes.append(box)

        ##################### REMOVE THIS ##############################
        # new_img = image.transpose((1, 2, 0))
        # new_img = new_img * 255
        # new_img = new_img.astype(np.uint8).copy()
        # print(old_img.shape, new_img.shape)
        # for box in boxes:
        #     print(box)
        #     cv2.rectangle(new_img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 3)

        # cv2.imwrite('old.png', old_img)
        # cv2.imwrite('new.png', new_img)
        # print('images saved')
        #################################################################

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
    # print((model.backbone.body))
    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    num_classes = 2  # 1 class (person) + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # print(model.roi_heads.box_predictor)
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
    # print(model)
    return model

def get_model_mobilenet(num_classes):
    # load a pre-trained model for classification and return
    # only the features
    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    # FasterRCNN needs to know the number of
    # output channels in a backbone. For mobilenet_v2, it's 1280
    # so we need to add it here
    
    backbone.out_channels = 1280
    # let's make the RPN generate 5 x 3 anchors per spatial
    # location, with 5 different sizes and 3 different aspect
    # ratios. We have a Tuple[Tuple[int]] because each feature
    # map could potentially have different sizes and
    # aspect ratios 
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))

    # let's define what are the feature maps that we will
    # use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling.
    # if your backbone returns a Tensor, featmap_names is expected to
    # be [0]. More generally, the backbone should return an
    # OrderedDict[Tensor], and in featmap_names you can choose which
    # feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                    output_size=7,
                                                    sampling_ratio=2)

    # put the pieces together inside a FasterRCNN model
    model = FasterRCNN(backbone,
                       num_classes=2,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)
    return model

def get_transform(train):
    transforms = []

    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    # if train:
    #     # during training, randomly flip the training images
    #     # and ground-truth for data augmentation
    #     transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

dataset = TableDataset('data', get_transform(train=True))
dataset_test = TableDataset('data', get_transform(train=False))

# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

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
# model = get_model_mobilenet(num_classes)

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
writer = SummaryWriter()
# let's train it for 10 epochs
num_epochs = 100
step = 0
# exit(0)
for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    #train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=100)
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 30
    # lr_scheduler = None

    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        
        loss_value = losses_reduced.item()
        
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        step += 1

        if (step % 100 == 0):
            for key, val in loss_dict.items():
                # all_losses[key] = val
            #     # if (summary is not None):
                writer.add_scalar(key, val.item(), step)
            # all_losses["total_loss"] = loss_value
            writer.add_scalar('loss: ', loss_value, step)

    # update the learning rate
    if lr_scheduler is not None:
        lr_scheduler.step()

    print('evaluating...')
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)

    if (epoch%9 == 0):
        torch.save(model.state_dict(), 'saved_model/model{}.pth'.format(epoch+1))
    
    torch.cuda.empty_cache()

writer.close()

# pick one image from the test set
img, _ = dataset_test[0]
# put the model in evaluation mode
model.eval()
with torch.no_grad():
    prediction = model([img.to(device)])

print(prediction)
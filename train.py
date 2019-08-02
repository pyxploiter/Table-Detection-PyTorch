import os
import math
import sys
import argparse

import cv2
import PIL
from PIL import Image

import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

import utils
from transforms import get_transform
from dataset import TableDataset
import models

parser = argparse.ArgumentParser()

parser.add_argument("-p", "--path", dest="train_path", help="Path to training data images.", default="data/train")
parser.add_argument("-l", "--label", dest="train_label", help="Path to training data labels.", default="data/train.csv")
parser.add_argument("--hf", dest="horizontal_flips", help="Augment with horizontal flips in training. (Default=false).", action="store_true", default=False)
parser.add_argument("-e","--num_epochs", type=int, dest="num_epochs", help="Number of epochs.", default=10)
parser.add_argument("--cf","--check_freq", type=int, dest="check_freq", help="Checkpoint frequency.")
parser.add_argument("-o","--output_weight_path", dest="output_weight_path", help="Output path for weights.", default='saved_model')
parser.add_argument("-i","--input_weight_path", dest="input_weight_path", help="Input path for weights.")

options = parser.parse_args()

output_weight_path = "saved_model"
check_freq = 10
num_epochs = 100

def init_msg():
    print('---------------------------------------------------------')
    print('[info] total epochs:', num_epochs)
    if check_freq:
        print('[info] model weights will saved in /'+output_weight_path+' folder after every', options.check_freq, 'epochs')
    else:
        print('[info] model weights will saved in /'+output_weight_path+' folder')
    print('---------------------------------------------------------')

def save_model_design(model, output_name="frcnn"):
    if not os.path.exists("models"):
        os.makedirs('models')
    f = open("models/"+output_name+'.json','w')
    f.write(str(model))
    f.close()

def load_weights(model, model_path):
    # if checkpoint saved, continue training from that checkpoin
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("model loaded from "+model_path)
    return model

def train_one_epoch(model, optimizer, data_loader, writer, device, epoch, print_freq):
    step = 0
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    wp_lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        wp_lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        step += 1
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

        if wp_lr_scheduler is not None:
            wp_lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        # write scalars to tensorboard
        if (step % print_freq == 0):
            for key, val in loss_dict.items():
                # adding 4 losses one by one
                writer.add_scalar(key, val.item(), step)
            # adding total loss
            writer.add_scalar('loss: ', loss_value, step)

dataset = TableDataset(os.getcwd(), get_transform(train=True))
dataset_test = TableDataset(os.getcwd(), get_transform(train=False))

# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
# dataset = torch.utils.data.Subset(dataset, indices[:-50])
dataset = torch.utils.data.Subset(dataset, indices)
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
model = models.frcnn_resnet101(num_classes)

# save_model_design(model)

# load_weights(model, "saved_model/model.pth")

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.SGD(params, lr=0.003,
                            momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by 10x
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=40,
                                               gamma=0.1)

# create the summary writer
writer = SummaryWriter()

init_msg()

for epoch in range(num_epochs):
  	# enter into training mode
    model.train()
    #train one epoch
    train_one_epoch(model, optimizer, data_loader, writer, device, epoch, print_freq=200)
    
    # update the learning rate after the step-size defined in LR scheduler
    if lr_scheduler.get_lr()[0] > 0.000001:
        lr_scheduler.step()

    # check if output_weight_path directory exists
    if not os.path.exists(output_weight_path):
        os.makedirs(output_weight_path)

    # if model checkpoints frequency is provided
    if check_freq:
        # save model weights at given frequency
        if ((epoch+1)%check_freq == 0):
            torch.save(model.state_dict(), output_weight_path+'/model_ep{}.pth'.format(epoch+1))
    # save the last weights 
    if ((epoch+1) == num_epochs):
        torch.save(model.state_dict(), output_weight_path+'/model_ep{}.pth'.format(epoch+1))

    torch.cuda.empty_cache()
print('[info] training is completed.')
writer.close()

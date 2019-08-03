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
from parser import params

parser = argparse.ArgumentParser()

parser.add_argument("-p", "--path", dest="train_images_path", help="Path to training data images.")
parser.add_argument("-l", "--label", dest="train_labels_path", help="Path to training data labels.")
parser.add_argument("-e","--num_epochs", type=int, dest="num_epochs", help="Number of epochs.")
parser.add_argument("--cf","--check_freq", type=int, dest="check_freq", help="Checkpoint frequency.")
parser.add_argument("-o","--output_weight_path", dest="output_weight_path", help="Output path for weights.", default='saved_model')
parser.add_argument("-i","--input_weight_path", dest="input_weight_path", help="Input path for weights.")

options = parser.parse_args()

# read command line arguments if given, otherwise read it from config file
train_images_path = options.train_images_path if options.train_images_path else params['train_images_path'] 
train_labels_path = options.train_labels_path if options.train_labels_path else params['train_labels_path']
num_epochs = options.num_epochs if options.num_epochs else params['num_epochs']
check_freq = options.check_freq if options.check_freq else params['check_freq']
output_weight_path = options.output_weight_path if options.output_weight_path else params['output_weight_path']
input_weight_path = options.input_weight_path if options.input_weight_path else params['input_weight_path']

def init_msg():
    print('---------------------------------------------------------')
    print('[info] training images path: '+ train_images_path)
    print('[info] training labels path: '+ train_labels_path)
    print('[info] total epochs:', num_epochs)
    if check_freq:
        print('[info] model weights will saved in /'+output_weight_path+' folder after every', options.check_freq, 'epochs')
    else:
        print('[info] model weights will saved in /'+output_weight_path+' folder')
    print('---------------------------------------------------------')

# saves model architecture in json file
def save_model_design(model, output_name="frcnn"):
    if not os.path.exists("models"):
        os.makedirs('models')
    f = open("models/"+output_name+'.json','w')
    f.write(str(model))
    f.close()

# loads model weights from file
def load_weights(model, model_path):
    # if checkpoint saved, continue training from that checkpoin
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("model loaded from "+model_path)
    return model

# for each epoch, this function is called
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

dataset = TableDataset(os.getcwd(), train_images_path, train_labels_path, get_transform(train=True))

# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()

dataset = torch.utils.data.Subset(dataset, indices[:20])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=True, num_workers=2,
    collate_fn=utils.collate_fn)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and table
num_classes = 2

# get the model using our helper function
model = models.frcnn_resnet50_fpn(num_classes)

# load weights
load_weights(model, input_weight_path)

# move model to the right device
model.to(device)

# construct an optimizer
model_params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.SGD(model_params, lr=params['learning_rate'],
                            momentum=params['momentum'], weight_decay=params['weight_decay'])

# and a learning rate scheduler which decreases the learning rate by 10x
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=params['step_size'],
                                               gamma=params['gamma'])

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

import os
import argparse

from skimage import transform as sktsf
import cv2
from PIL import Image
import numpy as np
import csv
import transforms as T
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
from torchvision import transforms as tvtsf
import models

import utils
from parser import params

parser = argparse.ArgumentParser()

parser.add_argument("-p", dest="test_images_path", help="Path to test data images.")
parser.add_argument("-c", dest="checkpoint_path", help="Input checkpoint file path.")

options = parser.parse_args()

# read command line arguments if given, otherwise read it from config file
test_images_path = options.test_images_path if options.test_images_path else params['test_images_path'] 
checkpoint_path = options.checkpoint_path if options.checkpoint_path else params['checkpoint_path']

num_classes = 2
model = models.frcnn_resnet50_fpn(num_classes)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if not os.path.exists(checkpoint_path):
    print(checkpoint_path + ' file does not exists')
    exit(0)
else:
    print("[info] loading model from "+ checkpoint_path)

# loading saved model weights
model.load_state_dict(torch.load(checkpoint_path))
# move model to the right device
model.to(device)
# evaluation mode ON
model.eval()

# check if test directory exist
if not os.path.exists(test_images_path):
    print(test_images_path + ' does not exists')
    exit(0)

test_images = os.listdir(test_images_path)

# check if evaluation folder exists, otherwise create it
if not os.path.exists('evaluation'):
    os.makedirs('evaluation')

with open('evaluation/predicted_bboxes.csv', 'wt') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    filewriter.writerow(['image_id', 'xmin', 'ymin', 'xmax', 'ymax', 'label', 'prob'])
    
    print('[info]', len(test_images),'images loaded for test.\n')
    count = 0
    for img_path in test_images:
        img = cv2.imread(os.path.join(test_images_path, img_path))
        
        # img = utils.distance_transform(img)
        img = img.astype('float32')

        if img.ndim == 2:
            # reshape (H, W) -> (1, H, W)
            immg = img[np.newaxis]
        else:
            # transpose (H, W, C) -> (C, H, W)
            img = img.transpose((2, 0, 1))
        C, H, W = img.shape
        img = utils.preprocess_image(img)
        
	    # read original image for printing bounding boxes on it
        image_to_write = cv2.imread(os.path.join(test_images_path, img_path))

        with torch.no_grad():
            prediction = model([img.to(device)])

        predicted_boxes = []
        predicted_scores = []
        
        for i in range(prediction[0]['boxes'].size()[0]):
            if (prediction[0]['scores'][i] > 0.6):
                box = prediction[0]['boxes'][i]
                xmin = int(box[0].item())
                ymin = int(box[1].item())
                xmax = int(box[2].item())
                ymax = int(box[3].item())
                predicted_boxes.append([xmin, ymin, xmax, ymax])
                predicted_scores.append(prediction[0]['scores'][i].item())
        
        # if any bbox is predicted
        if (predicted_boxes):    
            # Resize bboxes according to image resize
            _, o_H, o_W = img.shape
            bbox = np.stack(predicted_boxes).astype(np.float32)
            resized_boxes = utils.resize_bbox(bbox, (o_H, o_W), (H, W))

            # resized boxes are stacked (R, 4) 
            # where R is the number of bboxes in the image 
            # converted it back to simple 2d-array [[xmin1, ymin1, xmax1, ymax1], ...]
            boxes = []
            for i in resized_boxes:
                box = []
                [box.append(int(b)) for b in i]
                boxes.append(box)

            # printing bboxes on image and storing them in csv file
            for i in range(len(boxes)):
                cv2.rectangle(image_to_write, (boxes[i][0], 
                    boxes[i][1]), (boxes[i][2], boxes[i][3]), (0, 0, 255), 3)
                
                filewriter.writerow(
                    [
                        img_path,
                        boxes[i][0],
                        boxes[i][1],
                        boxes[i][2],
                        boxes[i][3],
                        'table',
                        predicted_scores[i]
                    ])
        else:
            filewriter.writerow(
                        [
                            img_path,
                            0,
                            0,
                            0,
                            0,
                            'no detections',
                            0
                        ])
        count += 1
        print("["+str(count)+"/"+str(len(test_images))+"] | image_id:", img_path)
        if not os.path.exists('predictions'):
            os.makedirs('predictions')
        cv2.imwrite('predictions/'+img_path, image_to_write)

print('[info] testing is completed.')

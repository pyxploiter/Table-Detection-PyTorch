import os
import argparse

from skimage import transform as sktsf
import cv2
from PIL import Image
import numpy as np
import csv
import transforms as T
import torch
import torch.utils.data
import torchvision
from torchvision import transforms as tvtsf
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import utils

parser = argparse.ArgumentParser()

parser.add_argument("-p", dest="test_path", help="Path to test data images.", default="data/test")
parser.add_argument("-c", dest="input_checkpoint", help="Input checkpoint file path.", required=True)

options = parser.parse_args()

def get_model_resnet(num_classes):
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
    # print(model)
    return model

num_classes = 2
model = get_model_resnet(num_classes)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# loading saved model weights
model.load_state_dict(torch.load(options.input_checkpoint))
# move model to the right device
model.to(device)
# evaluation mode ON
model.eval()

# check if test directory exist
if not os.path.exists(options.test_path):
    print(options.test_path + ' does not exists')
    exit(0)

test_dir = options.test_path
test_images = os.listdir(test_dir)

if not os.path.exists('evaluation'):
    os.makedirs('evaluation')

with open('evaluation/predictions.csv', 'wt') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    filewriter.writerow(['image_id', 'xmin', 'ymin', 'xmax', 'ymax', 'label', 'prob'])
    
    print('[info]', len(test_images),'images loaded for test.\n')
    count = 0
    for img_path in test_images:
        img = utils.read_image(os.path.join(test_dir, img_path))
        # Rescaling Images
        C, H, W = img.shape
        min_size = 600
        max_size = 1024
        scale1 = min_size / min(H, W)
        scale2 = max_size / max(H, W)
        scale = min(scale1, scale2)
        img = img / 255.
        img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect',anti_aliasing=False)

        # Normalizing image
        normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        img = normalize(torch.from_numpy(img))

        image_to_write = cv2.imread(os.path.join(test_dir, img_path))

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
        # if not os.path.exists('data/output'):
        #     os.makedirs('data/output')
        # cv2.imwrite('data/output/'+img_path, image_to_write)
        # print(img_path)
print('[info] testing is completed.')
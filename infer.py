import os
from skimage import transform as sktsf
import cv2
from PIL import Image
import numpy as np
import transforms as T
import torch
import torch.utils.data
import torchvision
from torchvision import transforms as tvtsf
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import utils

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

num_classes = 2
model = get_model_resnet(num_classes)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.load_state_dict(torch.load('saved_model/model37.pth'))
model.to(device)

model.eval()

test_dir = "data/test/"
test_images = os.listdir(test_dir)

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
    for i in range(prediction[0]['boxes'].size()[0]):
        if (prediction[0]['scores'][i] > 0.85):
            box = prediction[0]['boxes'][i]
            xmin = int(box[0].item())
            ymin = int(box[1].item())
            xmax = int(box[2].item())
            ymax = int(box[3].item())
            predicted_boxes.append([xmin, ymin, xmax, ymax])
    
    if (predicted_boxes):    
        _, o_H, o_W = img.shape
        bbox = np.stack(predicted_boxes).astype(np.float32)
        resized_boxes = utils.resize_bbox(bbox, (o_H, o_W), (H, W))

        boxes = []
        for i in resized_boxes:
            box = []
            [box.append(int(b)) for b in i]
            boxes.append(box)

        for box in boxes:
            cv2.rectangle(image_to_write, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 3)
    
    cv2.imwrite('output/'+img_path, image_to_write)
    print(img_path)
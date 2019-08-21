import os
import argparse

import cv2
from PIL import Image
import numpy as np
import csv
import transforms as T
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import pytesseract
from skimage import transform as sktsf
from torchvision import transforms as tvtsf
import models

import utils
from parser import params
import postprocess as postp


parser = argparse.ArgumentParser()

parser.add_argument("-p", dest="test_images_path", help="Path to test data images.")
parser.add_argument("-c", dest="checkpoint_path", help="Input checkpoint file path.")

options = parser.parse_args()

# read command line arguments if given, otherwise read it from config file
test_images_path = options.test_images_path if options.test_images_path else params['test_images_path'] 
checkpoint_path = options.checkpoint_path if options.checkpoint_path else params['checkpoint_path']

APPLY_NMS = True
APPLY_OCR = True
debug_level = 2

def constant_aspect_resize(image, width=2500, height=None, interpolation=None):
    """performs resizing of images according to the given width"""
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    if interpolation is not None:
        resized = cv2.resize(image, dim, interpolation=interpolation)
    elif dim[0] < w:
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    else:
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_LANCZOS4)
    return resized


class Rect:
    """internal custom class used to perform rectangular calculations used for evaluation"""

    def __init__(self, x1, y1, w, h, prob=0.0, text=None):
        x2 = x1 + w
        y2 = y1 + h
        if x1 > x2 or y1 > y2:
            raise ValueError("Coordinates are invalid")
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.w, self.h = w, h
        self.prob = prob
        self.text = text

    def area_diff(self, other):
        """calculates the area of box that is non-intersecting with 'other' rect"""
        return self.area() - self.intersection(other).area()

    def area(self):
        """calculates the area of the box"""
        return self.w * self.h

    def intersection(self, other):
        """calculates the intersecting area of two rectangles"""
        a, b = self, other
        x1 = max(a.x1, b.x1)
        y1 = max(a.y1, b.y1)
        x2 = min(a.x2, b.x2)
        y2 = min(a.y2, b.y2)
        if x1 < x2 and y1 < y2:
            return type(self)(x1, y1, x2 - x1, y2 - y1)
        else:
            return type(self)(0, 0, 0, 0)

    __and__ = intersection

    def union(self, other):
        """takes the union of the two rectangles"""
        a, b = self, other
        x1 = min(a.x1, b.x1)
        y1 = min(a.y1, b.y1)
        x2 = max(a.x2, b.x2)
        y2 = max(a.y2, b.y2)
        if x1 < x2 and y1 < y2:
            return type(self)(x1, y1, x2 - x1, y2 - y1)

    __or__ = union
    __sub__ = area_diff

    def __iter__(self):
        yield self.x1
        yield self.y1
        yield self.x2
        yield self.y2

    def __eq__(self, other):
        return isinstance(other, Rect) and tuple(self) == tuple(other)

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return type(self).__name__ + repr(tuple(self))


def compute_overlap(i: Rect, j: Rect):
    """computes the overlap between two rectangles in the range (0, 1)"""
    return 2 * abs((i & j).area()) / (abs(i.area() + j.area()) + 1)


def compute_contain(i: Rect, j: Rect):
    """compute the area ratio of smaller rectangle contained inside the other rectangle"""
    return abs((i & j).area()) / (min(i.area(), j.area()) + 1)


def remove_image_background(image):
    """this methood removes the background artifacts from the image
    background can include watermarks, borders and lines etc"""
    image2 = np.copy(image)
    kernel = np.ones((1, 5), np.uint8)
    lines1 = np.copy(image)
    lines1 = cv2.dilate(lines1, kernel, iterations=17)
    lines1 = cv2.erode(lines1, kernel, iterations=17)

    kernel = np.ones((5, 1), np.uint8)
    lines2 = np.copy(image)
    lines2 = cv2.dilate(lines2, kernel, iterations=17)
    lines2 = cv2.erode(lines2, kernel, iterations=17)

    lines2 = np.uint8(np.clip(np.int16(lines2) - np.int16(lines1) + 255, 0, 255))
    lines = np.uint8(
        np.clip((255 - np.int16(lines1)) + (255 - np.int16(lines2)), 0, 255)
    )

    bg_removed = np.uint8(np.clip(np.int16(image2) + np.int16(lines), 0, 255))

    return bg_removed


def perform_ocr(image):
    """performs ocr of image located at 'src' and saves the ocr data in 'dst' in pickle format"""
    org_image = constant_aspect_resize(image, width=2500)

    bg_removed = remove_image_background(np.copy(org_image))
    toOcr = cv2.GaussianBlur(cv2.cvtColor(bg_removed, cv2.COLOR_BGR2RGB), (3, 3), 0)
    ocr = pytesseract.image_to_data(
        Image.fromarray(toOcr), output_type=pytesseract.Output.DICT, lang='eng', config='--oem 1'
    )

    bboxes = []
    for i in range(len(ocr["conf"])):
        if ocr["level"][i] > 4:
            rect = Rect(
                ocr["left"][i] * image.shape[1] // 2500,
                ocr["top"][i] * image.shape[1] // 2500,
                ocr["width"][i] * image.shape[1] // 2500,
                ocr["height"][i] * image.shape[1] // 2500,
                float(ocr["conf"][i]),
                ocr["text"][i],
            )
            bboxes.append(rect)
    bboxes = sorted(bboxes, key=lambda x: x.area(), reverse=True)
    threshold = np.average(
        [box.area() for box in bboxes[len(bboxes) // 20 : -len(bboxes) // 4]]
    )
    threshold *= 30
    threshold = min(threshold, (image.shape[1] ** 2) // 2)
    bboxes = [box for box in bboxes if box.area() < threshold]
    return bboxes

def apply_nms(predictions, overlap_thresh=0.5, contain_thresh=0.75):
    """THIS METHOD PERFORMS THE NMS OPERATION ON ALL AVAILABLE PREDICTIONS,
    NMS = NON MAXIMA SUPRESSION"""
    predictions = sorted(predictions, key=lambda x: x.prob, reverse=True)
    i = 0
    while i < len(predictions):
        j = i + 1
        while j < len(predictions):
            if (
                compute_contain(predictions[i], predictions[j]) > contain_thresh
                or compute_overlap(predictions[i], predictions[j]) > overlap_thresh
            ):
                if (
                    2
                    * (predictions[i].prob - predictions[j].prob)
                    / (predictions[i].prob + predictions[j].prob)
                    > 0
                ):
                    predictions.remove(predictions[j])
                    continue
            j += 1
        i += 1
    return predictions

def bounding_box(rect, bboxes):
    x1 = 10000
    y1 = 10000
    x2 = 0
    y2 = 0

    for b in bboxes:
        x1 = min(b.x1, x1)
        y1 = min(b.y1, y1)
        x2 = max(b.x2, x2)
        y2 = max(b.y2, y2)

    rect = Rect(x1, y1, x2 - x1, y2 - y1, rect.prob, rect.text)
    return rect

def tight_fit(tables, text_boxes):
    """compute text_boxes contained inside each table on the basis of overlap ratio,
    and use them to get a tight fit of the 'tables' bounding boxes"""
    for i in range(len(tables)):
        boxes_contained = []
        for b in text_boxes:
            if compute_contain(b, tables[i]) > 0.5:
                boxes_contained.append(b)
        if len(boxes_contained) > 0:
            tables[i] = bounding_box(tables[i], boxes_contained)
    return tables


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
        image_to_write = cv2.imread(os.path.join(test_images_path, img_path))
        
        # img = utils.distance_transform(img)
        img = image_to_write.astype('float32')

        if img.ndim == 2:
            # reshape (H, W) -> (1, H, W)
            immg = img[np.newaxis]
        else:
            # transpose (H, W, C) -> (C, H, W)
            img = img.transpose((2, 0, 1))
        C, H, W = img.shape
        img = utils.preprocess_image(img)
        
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

            tables = []
            for score, box in zip(predicted_scores, boxes):
                tables.append(
                    Rect(
                        box[0],
                        box[1],
                        box[2] - box[0],
                        box[3] - box[1],
                        score
                    )
                )

            for table in tables:
                cv2.rectangle(image_to_write, (table.x1, 
                    table.y1), (table.x2, table.y2), (150, 150, 255), 3)

            if APPLY_NMS:
                tables = apply_nms(tables)

            if APPLY_OCR:
                ocr = perform_ocr(image_to_write)
                if debug_level > 1:
                    for j in ocr:
                        cv2.rectangle(
                            image_to_write, (j.x1, j.y1), (j.x1 + j.w, j.y1 + j.h), (20, 20, 0), 1
                        )

                tables = tight_fit(tables, ocr)


            # printing bboxes on image and storing them in csv file
            for table in tables:
                cv2.rectangle(image_to_write, (table.x1, 
                    table.y1), (table.x2, table.y2), (0, 0, 255), 3)
                
                filewriter.writerow(
                    [
                        img_path,
                        table.x1,
                        table.y1,
                        table.x2,
                        table.y2,
                        'table',
                        table.prob
                    ])
        # else:
        #     filewriter.writerow(
        #                 [
        #                     img_path,
        #                     0,
        #                     0,
        #                     0,
        #                     0,
        #                     'no detections',
        #                     0
        #                 ])
        count += 1
        print("["+str(count)+"/"+str(len(test_images))+"] | image_id:", img_path)
        if not os.path.exists('predictions'):
            os.makedirs('predictions')
        cv2.imwrite('predictions/'+img_path, image_to_write)

print('[info] testing is completed.')

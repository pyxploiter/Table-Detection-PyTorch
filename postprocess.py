import os
import cv2
import pickle
import numpy as np
import pytesseract
from PIL import Image

NMS_FLAG = False
OCR_FLAG = True
RESIZE_FLAG = False
RESIZE_WIDTH = 2500
image_source_dir = "../data/images"
ocr_data_dir = "../data/ocr"

#utility functions start
class Rect:
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
        return self.area() - self.intersection(other).area()

    def area(self):
        return self.w * self.h

    def intersection(self, other):
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
        return not (self == other)

    def __repr__(self):
        return type(self).__name__ + repr(tuple(self))

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

def perform_ocr(image, dst):
    """performs ocr of image located at 'src' and saves the ocr data in 'dst' in pickle format"""
    # image = cv2.imread(src)
    if RESIZE_FLAG:
        org_image = constant_aspect_resize(image, width=RESIZE_WIDTH)
        image = np.copy(org_image)

    bg_removed = remove_image_background(image)
    toOcr = cv2.GaussianBlur(cv2.cvtColor(bg_removed, cv2.COLOR_BGR2RGB), (3, 3), 0)

    ocr = pytesseract.image_to_data(
        Image.fromarray(toOcr), output_type=pytesseract.Output.DICT
    )

    with open(dst, "wb") as handle:
        pickle.dump(ocr, handle, protocol=pickle.HIGHEST_PROTOCOL)

def read_ocr(image, shape, ocr_file):
    if not os.path.isfile(ocr_file):
        perform_ocr(
            image,
            ocr_file,
        )

    with open(ocr_file, 'rb') as handle:
        ocr = pickle.load(handle)

    bboxes = []
    for i in range(len(ocr['conf'])):
        if ocr['level'][i] > 4:
            rect = Rect(
                ocr['left'][i] * RESIZE_WIDTH // shape[1],
                ocr['top'][i] * RESIZE_WIDTH // shape[1],
                ocr['width'][i] * RESIZE_WIDTH // shape[1],
                ocr['height'][i] * RESIZE_WIDTH // shape[1],
                float(ocr['conf'][i]),
                ocr['text'][i]
            )
            bboxes.append(rect)
    bboxes = sorted(bboxes, key=lambda x: x.area(), reverse=True)
    threshold = np.average([box.area() for box in bboxes[len(bboxes) // 20: -len(bboxes) // 4]])
    threshold *= 30
    threshold = min(threshold, (RESIZE_WIDTH)**2//2)
    bboxes = [box for box in bboxes if box.area() < threshold]
    return bboxes

def compute_overlap(i, j):
    return 2 * abs((i & j).area()) / abs(i.area() + j.area())

def compute_contain(i, j):
    return abs((i & j).area()) / min(i.area(), j.area())

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
    for i in range(len(tables)):
        boxes_contained = []
        for b in text_boxes:
            if compute_contain(b, tables[i]) > 0.5:
                boxes_contained.append(b)
        if len(boxes_contained) > 0:
            tables[i] = bounding_box(tables[i], boxes_contained)
    return tables

def nms(predictions, overlap_thresh=0.5, contain_thresh=0.75):
    predictions = sorted(predictions, key=lambda x: x.prob, reverse=True)
    i = 0
    removed = []
    while i < len(predictions):
        j = i + 1
        while j < len(predictions):
            if (predictions[i] & predictions[j]).area() > \
                    min(predictions[i].area(), predictions[j].area()) * contain_thresh \
                    or compute_overlap(predictions[i], predictions[j]) > overlap_thresh:
                if 2 * (predictions[i].prob - predictions[j].prob) / (predictions[i].prob + predictions[j].prob) > 0:
                    removed.append(predictions[j])
                    predictions.remove(predictions[j])
                    continue
            j += 1
        i += 1
    return predictions, removed

def post_process(predictions, ground_truth, key, image, shape):
    global RESIZE_FLAG, NMS_FLAG, OCR_FLAG, RESIZE_WIDTH
    ground_truth_bounding_boxes = []
    rcnn_bounding_boxes = []
    rcnn_removed_boxes = []

    if key in predictions.keys():
        if NMS_FLAG:
            rcnn_bounding_boxes, rcnn_removed_boxes = nms(predictions[key])
        else:
            rcnn_bounding_boxes = predictions[key]
        if not RESIZE_FLAG:
            tmp = RESIZE_WIDTH / shape[1]
            rcnn_bounding_boxes = [Rect(int(b.x1 * tmp), int(b.y1 * tmp), int(b.w * tmp), int(b.h * tmp), b.prob)
                                    for b in rcnn_bounding_boxes]
            if NMS_FLAG:
                rcnn_removed_boxes = [Rect(int(b.x1 * tmp), int(b.y1 * tmp), int(b.w * tmp), int(b.h * tmp), b.prob)
                                        for b in rcnn_removed_boxes]

    if key in ground_truth:
        ground_truth_bounding_boxes = ground_truth[key]
        if not RESIZE_FLAG:
            tmp = RESIZE_WIDTH / shape[1]
            ground_truth_bounding_boxes = [Rect(int(b.x1 * tmp), int(b.y1 * tmp), int(b.w * tmp), int(b.h * tmp))
                                            for b in ground_truth_bounding_boxes]

    for j in rcnn_bounding_boxes:
        cv2.rectangle(
            image, (j.x1, j.y1), (j.x1 + j.w, j.y1 + j.h), (150, 180, 255), max(1, image.shape[0] // 400)
        )

    if OCR_FLAG:
        ocr = read_ocr(image, shape, os.path.join(ocr_data_dir, key.replace('.png', '.pkl')))

        for j in ocr:
            cv2.rectangle(
                image, (j.x1, j.y1), (j.x1 + j.w, j.y1 + j.h), (20, 20, 0), 1
            )
        ground_truth_bounding_boxes = tight_fit(ground_truth_bounding_boxes, ocr)
        rcnn_bounding_boxes = tight_fit(rcnn_bounding_boxes, ocr)
        rcnn_removed_boxes = tight_fit(rcnn_removed_boxes, ocr)

    for j in rcnn_bounding_boxes:
        cv2.rectangle(
            image, (j.x1-5, j.y1-5), (j.x1 - 5 + j.w, j.y1 - 5 + j.h), (20, 50, 200), max(1, image.shape[0] // 400)
        )

    for j in rcnn_removed_boxes:
        cv2.rectangle(
            image, (j.x1, j.y1), (j.x1 + j.w, j.y1 + j.h), (120, 30, 30), max(1, image.shape[0] // 400)
        )

    for j in rcnn_removed_boxes:
        cv2.putText(
            image,
            str(j.prob),
            (j.x1, j.y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (120, 30, 30),
            4,
            cv2.LINE_AA,
        )

    for j in ground_truth_bounding_boxes:
        cv2.rectangle(
            image, (j.x1, j.y1), (j.x1 + j.w, j.y1 + j.h), (0, 255, 80), max(1, image.shape[0] // 400)
        )

    for j in rcnn_bounding_boxes:
        cv2.putText(
            image,
            str(j.prob),
            (j.x1, j.y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (50, 100, 255),
            4,
            cv2.LINE_AA,
        )

    return ground_truth_bounding_boxes, rcnn_bounding_boxes, rcnn_removed_boxes, image
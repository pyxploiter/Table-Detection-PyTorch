import os
import cv2
import csv
import glob
import pickle
import argparse
import pytesseract
import numpy as np
import pandas as pd
from PIL import Image
from xml.etree import ElementTree

RESIZE_WIDTH = 2500
args = None


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


def compute_contain(i: Rect, j: Rect):
    """compute the area ratio of smaller rectangle contained inside the other rectangle"""
    return abs((i & j).area()) / min(i.area(), j.area())


def compute_overlap(i: Rect, j: Rect):
    """computes the overlap between two rectangles in the range (0, 1)"""
    return 2 * abs((i & j).area()) / abs(i.area() + j.area())


def directory_maker(dir_list):
    """makes the directories in the list passsed to it"""
    for path in dir_list:
        if not os.path.exists(path):
            os.mkdir(path)
        else:
            continue


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


def read_ocr(image, width=2500):
    org_image = constant_aspect_resize(image, width=width)

    bg_removed = remove_image_background(org_image.copy())
    toOcr = cv2.GaussianBlur(cv2.cvtColor(bg_removed, cv2.COLOR_BGR2RGB), (3, 3), 0)
    ocr = pytesseract.image_to_data(
        Image.fromarray(toOcr), output_type=pytesseract.Output.DICT, config='--oem 1'
    )

    bboxes = []
    for i in range(len(ocr["conf"])):
        if ocr["level"][i] > 4:
            rect = Rect(
                ocr["left"][i] * image.shape[1] // width,
                ocr["top"][i] * image.shape[1] // width,
                ocr["width"][i] * image.shape[1] // width,
                ocr["height"][i] * image.shape[1] // width,
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


def read_csv_data(path):
    """this function reads the csv data and returns a dict
    the dict contains image names as keys and a list of bounding boxes"""
    raw = pd.read_csv(
        path,
        dtype={
            "image_id": str,
            "xmin": np.int32,
            "ymin": np.int32,
            "xmax": np.int32,
            "ymax": np.int32,
            "label": str,
        },
    )

    raw["image_id"] = raw["image_id"].apply(lambda x: ".".join(x.split(".")[:-1]))

    grouped = raw.groupby("image_id").groups

    for key in grouped.keys():
        indices = grouped[key]

        if "prob" in raw:
            grouped[key] = [
                Rect(
                    raw.loc[idx].xmin,
                    raw.loc[idx].ymin,
                    raw.loc[idx].xmax - raw.loc[idx].xmin,
                    raw.loc[idx].ymax - raw.loc[idx].ymin,
                    raw.loc[idx].prob,
                )
                for idx in indices
                if raw.loc[idx].label == "table"
            ]
    return grouped


def read_xml_data(path):
    filenames = glob.glob(os.path.join(path, "*.xml"))

    xml_list = []
    for xml_file in filenames:
        tree = ElementTree.parse(xml_file)
        root = tree.getroot()

        for obj in root.findall(".//object"):
            if "table" in obj[0].text:

                for elem in obj.findall(".//bndbox"):
                    value = (
                        xml_file.split("/")[-1].replace(".xml", ""),
                        int(elem[0].text),
                        int(elem[1].text),
                        int(elem[2].text),
                        int(elem[3].text),
                        "table",
                    )
                    xml_list.append(value)

    column_name = ["image_id", "xmin", "ymin", "xmax", "ymax", "label"]
    raw = pd.DataFrame(xml_list, columns=column_name)

    grouped = raw.groupby("image_id").groups

    for key in grouped.keys():
        indices = grouped[key]

        if "prob" in raw:
            grouped[key] = [
                Rect(
                    raw.loc[idx].xmin,
                    raw.loc[idx].ymin,
                    raw.loc[idx].xmax - raw.loc[idx].xmin,
                    raw.loc[idx].ymax - raw.loc[idx].ymin,
                    raw.loc[idx].prob,
                )
                for idx in indices
                if raw.loc[idx].label == "table"
            ]
        else:
            grouped[key] = [
                Rect(
                    raw.loc[idx].xmin,
                    raw.loc[idx].ymin,
                    raw.loc[idx].xmax - raw.loc[idx].xmin,
                    raw.loc[idx].ymax - raw.loc[idx].ymin,
                )
                for idx in indices
                if raw.loc[idx].label == "table"
            ]
    return grouped


def bounding_box(rect, bboxes):
    """provided a list of bounding boxes (bboxes),
    compute the rectangle that contains all of them."""
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


def post_process(predictions, ground_truth, key, image):
    """given a list of predictions and ground_truth bounding boxes,
    post-process them to make them consistent in scale and re-align them"""
    ground_truth_bounding_boxes = []
    rcnn_bounding_boxes = []

    if key in predictions.keys():
        rcnn_bounding_boxes = predictions[key]

    if key in ground_truth:
        ground_truth_bounding_boxes = ground_truth[key]

    if args.debug_level > 1 and args.images:
        for j in rcnn_bounding_boxes:
            cv2.rectangle(
                image,
                (j.x1, j.y1),
                (j.x1 + j.w, j.y1 + j.h),
                (150, 180, 255),
                max(1, image.shape[0] // 400),
            )

    if args.ocr and args.images:
        ocr = read_ocr(image)

        if args.debug_level > 1:
            for j in ocr:
                cv2.rectangle(
                    image, (j.x1, j.y1), (j.x1 + j.w, j.y1 + j.h), (20, 20, 0), 1
                )

        ground_truth_bounding_boxes = tight_fit(ground_truth_bounding_boxes, ocr)
        rcnn_bounding_boxes = tight_fit(rcnn_bounding_boxes, ocr)

    if args.debug_level > 0 and args.images:
        for j in rcnn_bounding_boxes:
            cv2.rectangle(
                image,
                (j.x1 - 5, j.y1 - 5),
                (j.x1 - 5 + j.w, j.y1 - 5 + j.h),
                (20, 50, 200),
                max(1, image.shape[0] // 400),
            )

    if args.debug_level > 0 and args.images:
        for j in ground_truth_bounding_boxes:
            cv2.rectangle(
                image,
                (j.x1, j.y1),
                (j.x1 + j.w, j.y1 + j.h),
                (0, 255, 80),
                max(1, image.shape[0] // 400),
            )

    if args.debug_level > 1 and args.images:
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

    return ground_truth_bounding_boxes, rcnn_bounding_boxes, image


def read_image(path):
    candidates = glob.glob(path + ".*")

    images = [
        x
        for x in candidates
        if x.endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"))
    ]
    if len(images) > 0:
        return cv2.imread(images[0], 1)
    else:
        return None


def evaluate(args):
    """computes evaluation-metrics as per the specifications of
    'Performance Evaluation and Benchmarking of Six-Page Segmentation Algorithms'"""
    eval_dir = args.output_path
    evaluation_file = os.path.join(eval_dir, "evaluations.csv")

    if args.debug_level > 0:
        image_target_dir = os.path.join(eval_dir, "images/")
        over_segmentation_dir = os.path.join(image_target_dir, "Over-Segmented/")
        under_segmentation_dir = os.path.join(image_target_dir, "Under-Segmented/")
        false_positive_dir = os.path.join(image_target_dir, "False-Positives/")
        overlapping_dir = os.path.join(image_target_dir, "Overlapping/")
        missed_dir = os.path.join(image_target_dir, "Missed/")
        partial_dir = os.path.join(image_target_dir, "Partial/")
        directory_maker(
            [
                eval_dir,
                image_target_dir,
                over_segmentation_dir,
                under_segmentation_dir,
                false_positive_dir,
                overlapping_dir,
                missed_dir,
                partial_dir,
            ]
        )
    else:
        directory_maker([eval_dir])

    print("|-----------Starting evaluations for the collected predictions-----------|")
    ground_truth = read_xml_data(args.ground_truth_path)
    predictions = read_csv_data(args.prediction_csv)

    max_thresh = 0.9
    min_thresh = 0.1

    correct = 0
    partial = 0
    missed = 0
    false_positive = 0
    under_segmented = 0
    size_rcnn = 0
    over_segmented = 0

    area_gt_total = 0.0
    area_tabular_precision = 0.0
    area_tabular_recall = 0.0
    area_output_total = 0.0

    total_gt_boxes = 0
    num_overlapped = 0

    for counter, key in enumerate(ground_truth.keys()):
        if counter % 20 == 0:
            print(counter, " images processed.")
        if (args.debug_level > 0 or args.ocr) and args.images:
            image = read_image(os.path.join(args.images, key))
            shape = image.shape[:2]
            if image is None:
                print(key)
                raise Exception("Image File not found.")
        else:
            image = None
            shape = None

        ground_truth_bounding_boxes, rcnn_bounding_boxes, image = post_process(
            predictions, ground_truth, key, image
        )

        size_rcnn += len(rcnn_bounding_boxes)

        assignments = []
        overlaps = []
        missed_gt = 0

        flag = False
        is_partial = False
        for i in ground_truth_bounding_boxes:
            max_overlap = -1
            max_index = -1

            for j, jb in enumerate(rcnn_bounding_boxes):
                new_overlap = compute_overlap(i, jb)

                if new_overlap == 0:
                    continue

                if new_overlap > max_overlap:
                    max_overlap = new_overlap
                    max_index = j

            if max_index != -1:
                assignments.append((i, rcnn_bounding_boxes[max_index]))
                overlaps.append(max_overlap)
            else:
                missed_gt += 1
                flag = True

        partial_tables = []
        for i in range(len(assignments)):
            if overlaps[i] >= max_thresh:
                correct += 1
            elif max_thresh > overlaps[i] > min_thresh or (
                assignments[i][0] & assignments[i][1]
            ).area() > 0.9 * min(assignments[i][0].area(), assignments[i][1].area()):
                partial += 1
                is_partial = True
                partial_tables.append(assignments[i])
            else:
                missed_gt += 1
                flag = True

            if args.debug_level > 1:
                cv2.putText(
                    image,
                    str(overlaps[i]),
                    (assignments[i][0].x2 - 50, assignments[i][0].y1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (150, 80, 255),
                    4,
                    cv2.LINE_AA,
                )

        missed += missed_gt

        total_gt_boxes += len(ground_truth_bounding_boxes)

        is_written = False
        if flag and args.debug_level > 0 and args.images:
            cv2.imwrite(missed_dir + key + ".png", image)
            is_written = True

        if is_partial and args.debug_level > 0 and args.images:
            cv2.imwrite(partial_dir + key + ".png", image)
            is_written = True

        # Over-Segmented
        flag = False
        for i in ground_truth_bounding_boxes:
            overlap_count = 0
            for j in rcnn_bounding_boxes:
                overlap = compute_overlap(i, j)
                if overlap > min_thresh or (i & j).area() > 0.9 * j.area():
                    overlap_count += 1
            if overlap_count >= 2:
                over_segmented += overlap_count - 1
                flag = True

        if flag and args.debug_level > 0 and args.images:
            cv2.imwrite(over_segmentation_dir + key + ".png", image)
            is_written = True

        flag = False
        # Under-Segmented
        for j in rcnn_bounding_boxes:
            overlap_count = 0
            for i in ground_truth_bounding_boxes:
                overlap = compute_overlap(i, j)
                if overlap > min_thresh or (i & j).area() > 0.9 * i.area():
                    overlap_count += 1
            if overlap_count >= 2:
                under_segmented += overlap_count - 1
                flag = True

        if flag and args.debug_level > 0 and args.images:
            cv2.imwrite(under_segmentation_dir + key + ".png", image)
            is_written = True

        flag = False
        # Overlapping
        for j in range(len(rcnn_bounding_boxes)):
            for i in range(j + 1, len(rcnn_bounding_boxes)):
                if (
                    compute_overlap(rcnn_bounding_boxes[i], rcnn_bounding_boxes[j])
                    > min_thresh
                ):
                    num_overlapped += 1
                    flag = True

        if flag and args.debug_level > 0 and args.images:
            cv2.imwrite(overlapping_dir + key + ".png", image)
            is_written = True

        flag = False
        # False Positives
        for i in rcnn_bounding_boxes:
            max_overlap = -1
            for j in ground_truth_bounding_boxes:
                overlap = compute_overlap(i, j)
                if overlap > max_overlap:
                    max_overlap = overlap
            if max_overlap <= min_thresh:
                false_positive += 1
                flag = True

        if flag and args.debug_level > 0 and args.images:
            cv2.imwrite(false_positive_dir + key + ".png", image)
            is_written = True

        # Area Precision
        for i in rcnn_bounding_boxes:
            overlap_sum = 0
            for j in range(len(ground_truth_bounding_boxes)):
                new_overlap = (i & ground_truth_bounding_boxes[j]).area()
                for k in range(j):
                    new_overlap -= (
                        i
                        & ground_truth_bounding_boxes[k]
                        & ground_truth_bounding_boxes[j]
                    ).area()
                overlap_sum += new_overlap
            area_tabular_precision += overlap_sum
            area_output_total += i.area()

        # Area recall
        for i in ground_truth_bounding_boxes:
            overlap_sum = 0
            for j in range(len(rcnn_bounding_boxes)):
                new_overlap = (i & rcnn_bounding_boxes[j]).area()
                for k in range(j):
                    new_overlap -= (
                        i & rcnn_bounding_boxes[k] & rcnn_bounding_boxes[j]
                    ).area()
                overlap_sum += new_overlap
            area_tabular_recall += overlap_sum
            area_gt_total += i.area()
        if not is_written and args.debug_level > 0 and args.images:
            cv2.imwrite(image_target_dir + key + ".png", image)

    with open(evaluation_file, "wt") as csvfile:
        filewriter = csv.writer(
            csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        filewriter.writerow(["Total Predicted boxes: ", size_rcnn])
        filewriter.writerow(["Total Ground Truth boxes: ", total_gt_boxes])
        filewriter.writerow(["Correct:", (correct / total_gt_boxes) * 100])
        filewriter.writerow(["Partial: ", (partial / total_gt_boxes) * 100])
        filewriter.writerow(["Missed: ", (missed / total_gt_boxes) * 100])
        filewriter.writerow(
            ["Overlapping Predictions: ", (num_overlapped / size_rcnn) * 100]
        )
        filewriter.writerow(
            ["Over-segmented: ", (over_segmented / total_gt_boxes) * 100]
        )
        filewriter.writerow(
            ["Under-segmented: ", (under_segmented / total_gt_boxes) * 100]
        )
        filewriter.writerow(["False positives: ", (false_positive / size_rcnn) * 100])
        filewriter.writerow(
            ["Area precision: ", (area_tabular_precision / area_output_total) * 100]
        )
        filewriter.writerow(
            ["Area recall: ", (area_tabular_recall / area_gt_total) * 100]
        )
        print("\n\nEvaluations have been successfully written in following file:")
        print(evaluation_file, "\n\n")
    print("Total Predicted boxes: ", size_rcnn)
    print("Total Ground Truth boxes: ", total_gt_boxes)
    print("Correct:", (correct / total_gt_boxes) * 100, "%")
    print("Partial: ", (partial / total_gt_boxes) * 100, "%")
    print("Missed: ", (missed / total_gt_boxes) * 100, "%")
    print("Overlapping Predictions: ", (num_overlapped / size_rcnn) * 100, "%")
    print("Over-segmented: ", (over_segmented / total_gt_boxes) * 100, "%")
    print("Under-segmented: ", (under_segmented / total_gt_boxes) * 100, "%")
    print("False positives: ", (false_positive / size_rcnn) * 100, "%")
    print("Area precision: ", (area_tabular_precision / area_output_total) * 100, "%")
    print("Area recall: ", (area_tabular_recall / area_gt_total) * 100, "%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "ground_truth_path",
        action="store",
        help="path to folder containing XML ground truth files.",
    )
    parser.add_argument(
        "prediction_csv", action="store", help="path to prediction CSV file."
    )
    parser.add_argument(
        "output_path", action="store", help="path to store evaluation results."
    )
    parser.add_argument(
        "-i",
        "--images",
        help="path of folder containing images. (to be used with '--ocr' or visualization with 'debug_level > 0')",
        default="/data/images",
    )
    parser.add_argument(
        "--ocr",
        help="whether OCR is to be computed for better box re-alignment."
        "(If address is not provided, prediction alignment will be skipped)",
        action="store_true",
    )
    parser.add_argument(
        "-d",
        "--debug_level",
        help="level of debugging the code should be running in, currently two levels 1 and 0",
        action="store",
        type=int,
        choices=[0, 1, 2],
        default=0,
    )
    args = parser.parse_args()
    evaluate(args)

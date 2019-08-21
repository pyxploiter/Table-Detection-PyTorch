import os
import cv2
import csv
import glob
import shutil
import numpy as np
import pandas as pd
import postprocess as postp

postp.OCR_FLAG = True

#FLAGS
FLAGS = {}
FLAGS["RESIZE_WIDTH"] = 2500
FLAGS["RESIZE_FLAG"] = False

#PATHS
PATHS = {}
PATHS["PREDICTION_FILE"] = "evaluation/predictions.csv"
PATHS["EVALUATION_FILE"] = "evaluation/evaluations.csv"
PATHS["GT_FILE"] = "../data/70-20-10/b/test.csv"
PATHS["IM_SOURCE"] = "../data/images"
PATHS["EVAL_DIR"] = "evaluation"
PATHS["IM_TARGET_DIR"] = "evaluation/images"
PATHS["PARTIAL_DIR"] = "evaluation/images/Partial"
PATHS["MISSED_DIR"] = "evaluation/images/Missed"
PATHS["OVERSEG_DIR"] = "evaluation/images/Over-Segmented"
PATHS["UNDERSEG_DIR"] = "evaluation/images/Under-Segmented"
PATHS["OVERLAPPING_DIR"] = "evaluation/images/Overlapping"
PATHS["FALSEPOS_DIR"] = "evaluation/images/False-Positives"

PATHS["DIRECTORY_LIST_EVAL"] = [
    PATHS["EVAL_DIR"],
    PATHS["IM_TARGET_DIR"],
    PATHS["OVERSEG_DIR"],
    PATHS["UNDERSEG_DIR"],
    PATHS["FALSEPOS_DIR"],
    PATHS["OVERLAPPING_DIR"],
    PATHS["MISSED_DIR"],
    PATHS["PARTIAL_DIR"]
]

def constant_aspect_resize(image, width=2500, height=None, interpolation=None):
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

def directory_maker(dir_list):
    for path in dir_list:
        if not os.path.exists(path):
            os.mkdir(path)
        else:
            continue

def read_csv_data(path):
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

    grouped = raw.groupby("image_id").groups

    for key in grouped.keys():
        indices = grouped[key]

        if "prob" in raw:
            grouped[key] = [
                postp.Rect(
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
                postp.Rect(
                    raw.loc[idx].xmin,
                    raw.loc[idx].ymin,
                    raw.loc[idx].xmax - raw.loc[idx].xmin,
                    raw.loc[idx].ymax - raw.loc[idx].ymin,
                )
                for idx in indices
                if raw.loc[idx].label == "table"
            ]
    return grouped

#evaluate specific functions start

def compute_overlap(i, j):
    return 2 * abs((i & j).area()) / abs(i.area() + j.area())

def compute_contain(i, j):
    return abs((i & j).area()) / min(i.area(), j.area())
    
def calc_metrics():
    global PATHS,FLAGS
    print(
        "|-----------Starting evaluations for the collected predictions-----------|"
    )
    ground_truth = read_csv_data(PATHS["GT_FILE"])
    predictions = read_csv_data(PATHS["PREDICTION_FILE"])

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

    for counter, key in enumerate(predictions.keys()):
        if counter % 20 == 0:
            print(counter, " images processed.")
        image = cv2.imread(os.path.join(PATHS["IM_SOURCE"], key), 1)
        shape = image.shape[:2]
        
        if not FLAGS["RESIZE_FLAG"]:
            image = constant_aspect_resize(image, width=FLAGS["RESIZE_WIDTH"])
        
        if image is None:
            print(key)
            raise Exception("Image File not found.")

        ground_truth_bounding_boxes, rcnn_bounding_boxes, rcnn_removed_boxes, image = postp.post_process(predictions,
                                                                                                        ground_truth,
                                                                                                        key, image, shape)


        size_rcnn += len(rcnn_bounding_boxes)

        assignments = []
        overlaps = []
        missed_gt = 0

        flag = False
        isPartial = False
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
            ).area() > 0.9 * min(
                assignments[i][0].area(), assignments[i][1].area()
            ):
                partial += 1
                isPartial = True
                partial_tables.append(assignments[i])
            else:
                missed_gt += 1
                flag = True

            cv2.putText(
                image,
                str(overlaps[i]),
                (assignments[i][0].x2-50, assignments[i][0].y1),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (150, 80, 255),
                4,
                cv2.LINE_AA,
            )

        missed += missed_gt

        total_gt_boxes += len(ground_truth_bounding_boxes)

        is_written = False
        if flag:
            cv2.imwrite(os.path.join(PATHS["MISSED_DIR"], key), image)
            is_written = True

        if isPartial:
            cv2.imwrite(os.path.join(PATHS["PARTIAL_DIR"], key), image)
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

        if flag:
            cv2.imwrite(os.path.join(PATHS["OVERSEG_DIR"], key), image)
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

        if flag:
            cv2.imwrite(os.path.join(PATHS["UNDERSEG_DIR"], key), image)
            is_written = True

        flag = False
        # Overlapping
        for j in range(len(rcnn_bounding_boxes)):
            for i in range(j + 1, len(rcnn_bounding_boxes)):
                if (
                    compute_overlap(
                        rcnn_bounding_boxes[i], rcnn_bounding_boxes[j]
                    )
                    > min_thresh
                ):
                    num_overlapped += 1
                    flag = True

        if flag:
            cv2.imwrite(os.path.join(PATHS["OVERLAPPING_DIR"], key), image)
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

        if flag:
            cv2.imwrite(os.path.join(PATHS["FALSEPOS_DIR"], key), image)
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
        if not is_written:
            cv2.imwrite(os.path.join(PATHS["IM_TARGET_DIR"], key), image)

    with open(PATHS["EVALUATION_FILE"], "wt") as csvfile:
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
        filewriter.writerow(
            ["False positives: ", (false_positive / size_rcnn) * 100]
        )
        filewriter.writerow(
            ["Area precision: ", (area_tabular_precision / area_output_total) * 100]
        )
        filewriter.writerow(
            ["Area recall: ", (area_tabular_recall / area_gt_total) * 100]
        )
        print("\n\nEvaluations have been successfully written in following file:")
        print(PATHS["EVALUATION_FILE"], "\n\n")
    print("Total Predicted boxes: ", size_rcnn)
    print("Total Ground Truth boxes: ", total_gt_boxes)
    print("Correct:", (correct / total_gt_boxes) * 100, "%")
    print("Partial: ", (partial / total_gt_boxes) * 100, "%")
    print("Missed: ", (missed / total_gt_boxes) * 100, "%")
    print("Overlapping Predictions: ", (num_overlapped / size_rcnn) * 100, "%")
    print("Over-segmented: ", (over_segmented / total_gt_boxes) * 100, "%")
    print("Under-segmented: ", (under_segmented / total_gt_boxes) * 100, "%")
    print("False positives: ", (false_positive / size_rcnn) * 100, "%")
    print(
        "Area precision: ", (area_tabular_precision / area_output_total) * 100, "%"
    )
    print("Area recall: ", (area_tabular_recall / area_gt_total) * 100, "%")

def eval():
    global FLAGS
    prediction_files = glob.glob(os.path.join(PATHS["EVAL_DIR"], "*.csv"))
    
    for file in prediction_files:
        root = file.replace(".csv", "")
        without_nms = os.path.join(root, "without-nms/")
        with_nms = os.path.join(root, "with-nms/")

        directory_maker([root, 
            without_nms, 
            with_nms])

        # PREDICTION_FILE = file
        PATHS["PREDICTION_FILE"] = file

        directory_maker(PATHS["DIRECTORY_LIST_EVAL"])
        postp.NMS_FLAG = False
        calc_metrics()

        shutil.move(PATHS["IM_TARGET_DIR"],
                    os.path.join(without_nms, "images/"))
        shutil.move(PATHS["EVALUATION_FILE"], os.path.join(without_nms, "evaluations.csv"))

        directory_maker(PATHS["DIRECTORY_LIST_EVAL"])
        postp.NMS_FLAG = True
        calc_metrics()

        shutil.move(PATHS["IM_TARGET_DIR"],
                    os.path.join(with_nms, "images/"))
        shutil.move(PATHS["EVALUATION_FILE"], os.path.join(with_nms, "evaluations.csv"))

        
        shutil.move(PATHS["PREDICTION_FILE"], os.path.join(root, file.split('/')[-1]))

eval()
import os
import torch
import utils
from torchvision import transforms as tvtsf
from skimage import transform as sktsf
import numpy as np
import pandas as pd

train_path = "../data/70-20-10/b/train"
train_label = "../data/70-20-10/b/train.csv"

class TableDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files
        self.imgs = list(sorted(os.listdir(os.path.join(root, train_path))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, train_path, self.imgs[idx])
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
        
        if not os.path.isfile(os.path.join(self.root, train_label)):
            print('[error]', train_label+' file not found')
            exit(0)

        train_labels = pd.read_csv(os.path.join(self.root, train_label))

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
    
        # Rescale bounding box
        _, o_H, o_W = image.shape
        scale = o_H / H
        bbox = np.stack(old_boxes).astype(np.float32)
        resized_boxes = utils.resize_bbox(bbox, (H, W), (o_H, o_W))
        
        # resized boxes are stacked (R, 4) 
        # where R is the number of bboxes in the image 
        # converted it back to simple 2d-array [[xmin1, ymin1, xmax1, ymax1], ...]
        boxes = []
        for i in resized_boxes:
            box = []
            [box.append(int(b)) for b in i]
            boxes.append(box)

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

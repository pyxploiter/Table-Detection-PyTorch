from __future__ import print_function

from collections import defaultdict, deque
import datetime
import pickle
import time
import errno
import os

import cv2
import numpy as np
from PIL import Image
from skimage import transform as sktsf
import torch
import torch.distributed as dist
from torchvision import transforms as tvtsf

def read_image(path, dtype=np.float32, color=True):
    """Read an image from a file

    Args:
        path (str): A path of image file.
        dtype: The type of array. The default value is :obj:`~numpy.float32`.
        color (bool): This option determines the number of channels.
            If :obj:`True`, the number of channels is three. In this case,
            the order of the channels is RGB. This is the default behaviour.
            If :obj:`False`, this function returns a grayscale image.

    Returns:
        ~numpy.ndarray: An image.
    """
    img = cv2.imread(path)
    img = img.astype('float32')

    if img.ndim == 2:
        # reshape (H, W) -> (1, H, W)
        return img[np.newaxis]
    else:
        # transpose (H, W, C) -> (C, H, W)
        return img.transpose((2, 0, 1))


def resize_bbox(bbox, in_size, out_size):
    """Resize bounding boxes according to image resize.

    The bounding boxes are expected to be packed into a two dimensional
    tensor of shape :math:`(R, 4)`, where :math:`R` is the number of
    bounding boxes in the image. The second axis represents attributes of
    the bounding box. They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`,
    where the four attributes are coordinates of the top left and the
    bottom right vertices.

    Args:
        bbox (~numpy.ndarray): An array whose shape is :math:`(R, 4)`.
            :math:`R` is the number of bounding boxes.
        in_size (tuple): A tuple of length 2. The height and the width
            of the image before resized.
        out_size (tuple): A tuple of length 2. The height and the width
            of the image after resized.

    Returns:
        ~numpy.ndarray:
        Bounding boxes rescaled according to the given image shapes.

    """
    bbox = bbox.copy()
    y_scale = float(out_size[0]) / in_size[0]
    x_scale = float(out_size[1]) / in_size[1]
    bbox[:, 0] = y_scale * bbox[:, 0]
    bbox[:, 2] = y_scale * bbox[:, 2]
    bbox[:, 1] = x_scale * bbox[:, 1]
    bbox[:, 3] = x_scale * bbox[:, 3]
    return bbox

def preprocess_image(image):
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

    return image

def distance_transform(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    b = cv2.distanceTransform(img, distanceType=cv2.DIST_L2, maskSize=3)
    g = cv2.distanceTransform(img, distanceType=cv2.DIST_L1, maskSize=3)
    r = cv2.distanceTransform(img, distanceType=cv2.DIST_C, maskSize=3)

    # merge the transformed channels back to an image
    return cv2.merge((b, g, r))
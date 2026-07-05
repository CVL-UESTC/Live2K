import random

import cv2
import numpy as np


def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False):
    """Apply random flip/rotation augmentation to images and optional flows."""
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment_img(img):
        if hflip:
            img = cv2.flip(img, 1)
        if vflip:
            img = cv2.flip(img, 0)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:
            flow = cv2.flip(flow, 1)
            flow[:, :, 0] *= -1
        if vflip:
            flow = cv2.flip(flow, 0)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if not isinstance(imgs, list):
        imgs = [imgs]
        single_img = True
    else:
        single_img = False

    imgs = [_augment_img(img) for img in imgs]

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
            single_flow = True
        else:
            single_flow = False
        flows = [_augment_flow(flow) for flow in flows]
        if single_flow:
            flows = flows[0]

    if single_img:
        imgs = imgs[0]

    if return_status:
        status = (hflip, vflip, rot90)
        return (imgs, flows, status) if flows is not None else (imgs, status)

    return (imgs, flows) if flows is not None else imgs

import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils import data as data

from data.data_util import paired_paths_from_new_folder
from data.transforms import augment
from utils import FileClient, imfrombytes, img2tensor


def resize_image(img, h, w):
    img = img * 255
    img = Image.fromarray(cv2.cvtColor(img.astype(np.uint8),
                                       cv2.COLOR_BGR2RGB))
    img = img.resize((w, h), Image.BICUBIC)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img.astype(np.float32) / 255.


class SRDataset(data.Dataset):
    """Dataset for Live Photo frame enhancement.

    Each sample folder should contain:
        gt.png
        ref.png
        lq_sequence/*.png with exactly 9 frames
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.io_backend_opt = opt['io_backend']
        self.file_client = None
        self.dataroot = opt['dataroot']
        self.paths = paired_paths_from_new_folder(self.dataroot)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        path_dict = self.paths[index]
        lq_paths = path_dict['lq_paths']
        gt_path = path_dict['gt_path']
        ref_path = path_dict['ref_path']
        sample_name = os.path.basename(os.path.dirname(gt_path))

        img_gt = imfrombytes(self.file_client.get(gt_path, 'gt'),
                             float32=True)
        img_ref = imfrombytes(self.file_client.get(ref_path, 'ref'),
                              float32=True)

        img_lqs = []
        for lq_path in lq_paths:
            img_lq = imfrombytes(self.file_client.get(lq_path, 'lq'),
                                 float32=True)
            img_lqs.append(img_lq)

        if self.opt['phase'] == 'train':
            all_imgs = [img_gt, img_ref] + img_lqs
            all_imgs = augment(all_imgs, self.opt['use_flip'],
                               self.opt['use_rot'])
            img_gt = all_imgs[0]
            img_ref = all_imgs[1]
            img_lqs = all_imgs[2:]

        gt_h, gt_w = img_gt.shape[:2]
        img_gt_low = resize_image(img_gt, gt_h // 4, gt_w // 4)

        img_gt = img2tensor(img_gt, bgr2rgb=True, float32=True)
        img_lqs = img2tensor(img_lqs, bgr2rgb=True, float32=True)
        img_ref = img2tensor(img_ref, bgr2rgb=True, float32=True)
        img_gt_low = img2tensor(img_gt_low, bgr2rgb=True, float32=True)

        return {
            'lq': torch.cat(img_lqs, dim=0),
            'gt': img_gt,
            'gt_low': img_gt_low,
            'ref': img_ref,
            'lq_path': sample_name,
            'gt_path': sample_name,
        }

    def __len__(self):
        return len(self.paths)

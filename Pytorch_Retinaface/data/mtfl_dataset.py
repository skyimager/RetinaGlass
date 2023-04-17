import os
import os.path
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import pandas as pd

class MTFLDataset(data.Dataset):
    def __init__(self, annotations_file, img_dir, glass_pad=10, preproc=None):
        self.annotations = pd.read_csv(annotations_file, header=None, delim_whitespace=True, skipinitialspace=True)
        self.img_dir = img_dir
        self.glass_pad = glass_pad
        self.preproc = preproc

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, *self.annotations.iloc[idx, 0].split("\\"))
        img = cv2.imread(img_path)
        height, width, _ = img.shape

        annotation = np.zeros((1, 15))
        # bbox (for glasses)
        annotation[0, 0] = float(self.annotations.iloc[idx, 1]) - float(self.glass_pad)  # label[0]  # x1
        annotation[0, 1] = float(self.annotations.iloc[idx, 6]) - float(self.glass_pad)  # y1
        annotation[0, 2] = float(self.annotations.iloc[idx, 2]) + float(self.glass_pad)  # label[0] + label[2]  # x2
        annotation[0, 3] = float(self.annotations.iloc[idx, 7]) + float(self.glass_pad)  # label[1] + label[3]  # y2

        # landmarks (5 keypoint on face)
        annotation[0, 4] = self.annotations.iloc[idx, 1]    # l0_x
        annotation[0, 5] = self.annotations.iloc[idx, 6]    # l0_y
        annotation[0, 6] = self.annotations.iloc[idx, 2]    # l1_x
        annotation[0, 7] = self.annotations.iloc[idx, 7]    # l1_y
        annotation[0, 8] = self.annotations.iloc[idx, 3]   # l2_x
        annotation[0, 9] = self.annotations.iloc[idx, 8]   # l2_y
        annotation[0, 10] = self.annotations.iloc[idx, 4]  # l3_x
        annotation[0, 11] = self.annotations.iloc[idx, 9]  # l3_y
        annotation[0, 12] = self.annotations.iloc[idx, 5]  # l4_x
        annotation[0, 13] = self.annotations.iloc[idx, 10]  # l4_y
        
        if (self.annotations.iloc[idx, -2]==1):
            annotation[0, 14] = 1 #glass present
        elif (self.annotations.iloc[idx, -2]==2):
            annotation[0, 14] = -1 #glass absent
        else:
            print("Wrong index chosen for Glasses as values other than 1/2 seen")
            
        target = np.array(annotation)
        if self.preproc is not None:
            img, target = self.preproc(img, target)

        return torch.from_numpy(img), target

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)

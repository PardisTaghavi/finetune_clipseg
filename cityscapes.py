import os
import numpy as np
import torch
import cv2
import glob
from torch.utils.data import Dataset
import albumentations as A
from labels import labels

id2trainId = {label.id: label.trainId for label in labels}

num_to_color = {
    0: [128, 64, 128],
    1: [244, 35, 232],
    2: [70, 70, 70],
    3: [102, 102, 156],
    4: [190, 153, 153],
    5: [153, 153, 153],
    6: [250, 170, 30],
    7: [220, 220, 0],
    8: [107, 142, 35],
    9: [152, 251, 152],
    10: [70, 130, 180]
}

class Cityscapes(Dataset):
    def __init__(self, root_dir, split='train_extra'):
        self.root_dir = root_dir
        self.split = split
        self.H = 798
        self.W = 798
        self.train_transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomResizedCrop(self.H, self.W)
        ])
    
        if self.split in ['train_extra']:
            self.pseudo_dir = os.path.join(self.root_dir, 'gCLIP', self.split)
            self.pseudo_files = glob.glob(os.path.join(self.pseudo_dir, '*/*'))
            self.pseudo_files.sort()
            # Convert pseudo file paths to image file paths
            self.image_files = [file.replace("gclip_color", "leftImg8bit") for file in self.pseudo_files]
            self.image_files = [file.replace("gCLIP", "leftImg8bit") for file in self.image_files]
            self.image_files.sort()

        elif self.split in ['val', 'train', 'test']:
            # Use the correct directories and file patterns
            self.image_dir = os.path.join(self.root_dir, 'leftImg8bit', self.split)
            self.label_dir = os.path.join(self.root_dir, 'gtFine', self.split)
            
            self.image_files = glob.glob(os.path.join(self.image_dir, '*/*_leftImg8bit.png'))
            self.label_files = glob.glob(os.path.join(self.label_dir, '*/*_gtFine_labelIds.png'))
            self.color_files = glob.glob(os.path.join(self.label_dir, '*/*_gtFine_color.png'))
            
            self.image_files.sort()
            self.label_files.sort()
            self.color_files.sort()
            
            # Debugging: Print paths and counts
            print(f"Image directory: {self.image_dir}")
            print(f"Label directory: {self.label_dir}")
            print(f"Found {len(self.image_files)} images and {len(self.label_files)} labels for {split} split")
            
            if len(self.image_files) > 0 and len(self.label_files) > 0:
                img_basename = os.path.basename(self.image_files[0]).replace('_leftImg8bit.png', '')
                lbl_basename = os.path.basename(self.label_files[0]).replace('_gtFine_labelIds.png', '')
                print(f"Sample image: {os.path.basename(self.image_files[0])}")
                print(f"Sample label: {os.path.basename(self.label_files[0])}")

        self.stuff_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.instace_classes = [11, 12, 13, 14, 15, 16, 17, 18, 19]
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        assert os.path.exists(self.image_files[idx]), f"Image file does not exist: {self.image_files[idx]}"
        file_name = os.path.basename(self.image_files[idx])
        image = cv2.imread(self.image_files[idx], cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to load image: {self.image_files[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # H, W, C
        cropped_image = image[0:970, 0:2048]
        image = cv2.resize(cropped_image, (2048, 1024), interpolation=cv2.INTER_LINEAR)
        image = np.array(image)

        if self.split in ['train_extra']:
            pseudo_path = self.pseudo_files[idx]
            assert os.path.exists(pseudo_path), f"Pseudo label file does not exist: {pseudo_path}"
            label_rgb = cv2.imread(pseudo_path, cv2.IMREAD_COLOR)
            if label_rgb is None:
                raise ValueError(f"Failed to load pseudo label image: {pseudo_path}")
            label_rgb = cv2.cvtColor(label_rgb, cv2.COLOR_BGR2RGB)  # H, W, C
            # Initialize label with default value 11 (for instance classes)
            label = torch.ones(1024, 2048, dtype=torch.long) * 11
            for i in range(11):
                color = np.array(num_to_color[i])
                mask = (label_rgb == color).all(axis=-1)
                label[mask] = i
            label = label.numpy()

        elif self.split in ['train', 'val']:
            label_path = self.label_files[idx]
            assert os.path.exists(label_path), f"Label file does not exist: {label_path}"
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # H, W
            if label is None:
                raise ValueError(f"Failed to load label image: {label_path}")
            
            # Map IDs to train IDs
            label = np.vectorize(id2trainId.get)(label)
            
            # Resize if necessary (to 1024x2048)
            if label.shape != (1024, 2048):
                label = cv2.resize(label, (2048, 1024), interpolation=cv2.INTER_NEAREST)
                
            # Set instance classes to 11
            for i in self.instace_classes:
                label[label == i] = 11

        # Convert to tensors: image as [3, H, W] and label as [H, W]
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        label = torch.from_numpy(label).long()
        return image, label

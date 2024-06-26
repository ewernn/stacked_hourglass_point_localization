import os
import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from multiprocessing import Pool
from tqdm import tqdm
import scipy.ndimage as ndimage

# Assuming TF refers to torchvision.transforms.functional
import torchvision.transforms.functional as TF
import torch.nn.functional as F

def image_open_bw(img_path):
    with Image.open(img_path) as img:
        return img.convert('L')

class CoordinateDataset(Dataset):
    def __init__(self, root_dir, csv, im_sz, output_res, augment=False, num_workers=32, only10=False, testing=False):
        self.root_dir = root_dir
        self.im_sz = im_sz
        self.output_res = output_res
        self.augment = augment
        self.testing = testing
        csv_path = f"Data_{csv}.csv"
        csv_file = os.path.join(root_dir, csv_path)
        self.data_frame = pd.read_csv(csv_file, header=0).head(10) if only10 else pd.read_csv(csv_file, header=0)

        image_paths = [os.path.join(self.root_dir, img_name) for img_name in self.data_frame.iloc[:, 0]]
        with Pool(num_workers) as pool:
            self.images = list(tqdm(pool.imap(image_open_bw, image_paths), total=len(image_paths)))

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        image = self.images[idx]
        points = self.data_frame.iloc[idx, 1:].values.astype('float').reshape(-1, 2)
        
        # Filter out NaN values
        valid_points = points[~np.isnan(points).any(axis=1)]
        
        if self.augment:
            image, valid_points = custom_transform(image, valid_points)

        image_tensor = transforms.Compose([
            transforms.ToTensor(),
        ])(image)

        heatmaps = self.generate_heatmaps(valid_points, self.output_res)

        if self.testing: return image_tensor, valid_points

        return image_tensor, heatmaps

    def generate_heatmaps(self, points, output_res):
        num_keypoints = len(points)
        heatmaps = np.zeros((num_keypoints, output_res, output_res), dtype=np.float32)
        for i, point in enumerate(points):
            x, y = int(point[0] * output_res), int(point[1] * output_res)
            if 0 <= x < output_res and 0 <= y < output_res:
                heatmaps[i, y, x] = 1
                heatmaps[i] = ndimage.gaussian_filter(heatmaps[i], sigma=1)
        return torch.tensor(heatmaps, dtype=torch.float32)

def custom_transform(image, points, degree_range=(-15, 15), translate_range=(0.1, 0.1), scale_range=(0.8, 1.2)):
    angle = random.uniform(*degree_range)
    translations = (random.uniform(-translate_range[0], translate_range[0]) * image.width,
                    random.uniform(-translate_range[1], translate_range[1]) * image.height)
    scale = random.uniform(*scale_range)

    transformed_image = TF.affine(image, angle=angle, translate=translations, scale=scale, shear=0)

    # Update points according to transformations
    cos_a, sin_a = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    # Apply rotation matrix to each point individually
    transformed_points = np.zeros_like(points)
    for i, point in enumerate(points):
        shifted_point = (point - 0.5) * scale
        rotated_point = np.dot(shifted_point, rotation_matrix)
        transformed_points[i] = rotated_point + 0.5 + np.array(translations) / np.array([image.width, image.height])

    transformed_points = np.clip(transformed_points, 0, 1)
    transformed_image = transforms.ColorJitter(contrast=(0.8, 1.2))(transformed_image)

    return transformed_image, transformed_points

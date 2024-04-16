import os
import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from multiprocessing import Pool
from tqdm import tqdm
import scipy.ndimage as ndimage

# Assuming TF refers to torchvision.transforms.functional
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torchvision.utils as vutils

def image_open_bw(img_path):
    with Image.open(img_path) as img:
        return img.convert('L')

class CoordinateDataset(Dataset):
    def __init__(self, root_dir, im_sz, output_res, augment=False, num_workers=32, only10=False, testing=False):
        self.root_dir = root_dir
        self.im_sz = im_sz
        self.output_res = output_res
        self.augment = augment
        self.testing = testing
        csv_file = os.path.join(root_dir, 'Kept_Data2.csv')
        self.data_frame = pd.read_csv(csv_file, header=0).head(10) if only10 else pd.read_csv(csv_file, header=0)

        image_paths = [os.path.join(self.root_dir, img_name) for img_name in self.data_frame.iloc[:, 0]]
        with Pool(num_workers) as pool:
            self.images = list(tqdm(pool.imap(image_open_bw, image_paths), total=len(image_paths)))

    # def __getitem__(self, idx):
    #     # Retrieve the pre-loaded image and corresponding points for the given index
    #     image = self.images[idx]
    #     points = self.data_frame.iloc[idx, 1:].values.astype('float').reshape(-1, 2)
    #     # Augment both the image and the points if augmentation is enabled
    #     if self.augment:
    #         image, points = custom_transform(image, points)
    #     # Convert the image to a tensor without unnecessary transformations
    #     image_tensor = TF.to_tensor(image)
    #     # Generate heatmaps from the provided points
    #     heatmaps = self.generate_heatmaps(points, self.output_res)
    #     #save_image(image_tensor, os.path.join(self.root_dir, 'exps', f"image_{idx}.png"))
    #     save_image(image_tensor, f"/home/eawern/Eq/stacked_hourglass_point_localization/exps/eq_exps/image_{idx}.png")
    #     # Return a tuple of the image tensor and either points or heatmaps depending on the mode
    #     return (image_tensor, points) if self.testing else (image_tensor, heatmaps)
    def __getitem__(self, idx):
        # Existing code to get image and points
        image = self.images[idx]
        points = self.data_frame.iloc[idx, 1:].values.astype('float').reshape(-1, 2)
        if self.augment:
            image, points = custom_transform(image, points)
        image_tensor = TF.to_tensor(image)
        heatmaps = self.generate_heatmaps(points, self.output_res)
        # Existing code to save the image
        # image_save_path = f"/home/eawern/Eq/stacked_hourglass_point_localization/exps/eq_exps/image_{idx}.png"
        # save_image(image_tensor, image_save_path)
        # # New code to save heatmaps
        # for i, heatmap in enumerate(heatmaps):
        #     heatmap_save_path = f"/home/eawern/Eq/stacked_hourglass_point_localization/exps/eq_exps/heatmap_{idx}_{i}.png"
        #     # Add channel dimension to heatmap before saving
        #     heatmap = heatmap.unsqueeze(0)  # From (H, W) to (C, H, W) with C=1
        #     vutils.save_image(heatmap, heatmap_save_path)

        return (image_tensor, points) if self.testing else (image_tensor, heatmaps)

    def __len__(self):
        # Return the length of the dataset
        return len(self.images)

    def generate_heatmaps(self, points, output_res):
        num_keypoints = len(points)
        heatmaps = np.zeros((num_keypoints, output_res, output_res), dtype=np.float32)
        for i in range(num_keypoints):
            x, y = int(points[i, 0] * output_res), int(points[i, 1] * output_res)
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

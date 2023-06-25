#!venv/bin/python3

"""
Contains functions for calculating Structural similarity index between ground truth and predicted wounds.

(c) All rights reserved.
original authors: Francisco M. Garcia-Moreno, Miguel Ángel Gutiérrez-Naranjo. 2023.

Source code:
https://github.com/frangam/wound-healing

Please see LICENSE.md for the full license document:
https://github.com/frangam/wound-healing/LICENSE.md
"""
import os
import skimage.io
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import argparse

def calculate_ssim(image_dir="results/next-image/synthetic/original/", target_dir="results/next-image/synthetic/prediction/"):
    image_files = sorted(os.listdir(image_dir))
    target_files = sorted(os.listdir(target_dir))
    ssim_scores = []

    for image_file, target_file in tqdm(zip(image_files, target_files), total=len(image_files), desc="Calculating SSIM"):
        image_path = os.path.join(image_dir, image_file)
        target_path = os.path.join(target_dir, target_file)

        image = skimage.io.imread(image_path)
        target = skimage.io.imread(target_path)

        # Convertir las imágenes a escala de grises si es necesario
        if image.ndim > 2:
            image = skimage.color.rgb2gray(image[..., :3])
        if target.ndim > 2:
            target = skimage.color.rgb2gray(target[..., :3])

        # Comprobar si las imágenes están normalizadas
        if image.min() >= 0 and image.max() <= 1 and target.min() >= 0 and target.max() <= 1:
            data_range = 1
        else:
            data_range = image.max() - image.min()

        # Calcular el índice SSIM
        score, diff = ssim(image, target, full=True, data_range=data_range)
        ssim_scores.append(score)

    mean_ssim = sum(ssim_scores) / len(ssim_scores)
    print("\nmean_ssim", mean_ssim)
    return mean_ssim




p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument('--base', type=str, default="results/next-image/real/", help='the base directory path')
p.add_argument('--source', type=str, default="original", help='the source directory name')
p.add_argument('--target', type=str, default="prediction", help='the target directory name')

args = p.parse_args()

base_path = args.base

image_dir = f"{base_path}{args.source}/"
target_dir = f"{base_path}{args.target}/"
mean_ssim = calculate_ssim(image_dir, target_dir)

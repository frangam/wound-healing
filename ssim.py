#!venv/bin/python3

"""
Contains functions for calculating Structural similarity index between ground truth and predicted wounds.

(c) All rights reserved.
original authors: Francisco M. Garcia-Moreno. 2023.

Source code:
https://github.com/frangam/wound-healing

Please see LICENSE.md for the full license document:
https://github.com/frangam/wound-healing/LICENSE.md
"""
import os
import numpy as np
import skimage.io
import skimage.color
from skimage.metrics import structural_similarity as ssim, mean_squared_error as mse
from scipy.linalg import sqrtm
from tqdm import tqdm
import argparse
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

import woundhealing as W


# load the Inception v3 model
inception_model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

def calculate_metrics(image_dir="results/next-image/synthetic/original/", target_dir="results/next-image/synthetic/prediction/"):
    image_files = sorted(os.listdir(image_dir))
    target_files = sorted(os.listdir(target_dir))
    ssim_scores = []
    mse_scores = []
    pl_scores = []
    image_activations = []
    target_activations = []

    for image_file, target_file in tqdm(zip(image_files, target_files), total=len(image_files), desc="Calculating metrics"):
        image_path = os.path.join(image_dir, image_file)
        target_path = os.path.join(target_dir, target_file)
        print("image_path", image_path, "target_path", target_path)

        image = skimage.io.imread(image_path)
        target = skimage.io.imread(target_path)
        print(image.shape)
        print(target.shape)
        print(np.unique(image))
        print(np.unique(target))   

        # # Convert the images to grayscale if necessary
        # if image.ndim > 2:
        #     image = skimage.color.rgb2gray(image[..., :3])
        # if target.ndim > 2:
        #     target = skimage.color.rgb2gray(target[..., :3])
        # print(image.shape)
        # print(target.shape)


        # Normalize the images to a range of 0 to 1
        image = image / 255.0
        # image = (image - image.min()) / (image.max() - image.min())
        target = target / 255.0
        # target = (target - target.min()) / (target.max() - target.min())

        print(np.unique(image))
        print(np.unique(target))    

        print(image.shape)
        print(target.shape)

        
        # Calculate the SSIM index
        score, diff = ssim(image, target, full=True, data_range=1.0)
        ssim_scores.append(score)
        print(score)  # Print SSIM or MSE score


        # Calculate the MSE
        score = mse(image, target)
        mse_scores.append(score)
        print(score)  # Print SSIM or MSE score



        # # Calculate the perceptual loss
        # image = preprocess_input(skimage.io.imread(image_path))
        # target = preprocess_input(skimage.io.imread(target_path))
        # print(image.shape)

        # # If the images are not RGB, convert them to RGB
        # if len(image.shape)<3 or image.shape[2] == 1:
        #     image = np.repeat(image, 3, axis=2)
        # if len(target.shape)<3 or target.shape[2] == 1:
        #     target = np.repeat(target, 3, axis=2)

        # Resize images to 299x299 since InceptionV3 expects images of this size
        # image = tf.image.resize(image, (299, 299))
        # target = tf.image.resize(target, (299, 299))
        # image = image[..., :3]  # Mantener solo los 3 primeros canales RGB
        # target = target[..., :3]  # Mantener solo los 3 primeros canales RGB


        # image_activations.append(inception_model.predict(image[np.newaxis]))
        # target_activations.append(inception_model.predict(target[np.newaxis]))
        # pl_scores.append(mse(image_activations[-1], target_activations[-1]))

    mean_ssim = sum(ssim_scores) / len(ssim_scores)
    mean_mse = sum(mse_scores) / len(mse_scores)
    # mean_pl = sum(pl_scores) / len(pl_scores)

    # Calculate the FID
    # image_activations = np.concatenate(image_activations)
    # target_activations = np.concatenate(target_activations)
    # mu1 = image_activations.mean(axis=0)
    # mu2 = target_activations.mean(axis=0)
    # sigma1 = np.cov(image_activations, rowvar=False)
    # sigma2 = np.cov(target_activations, rowvar=False)
    # diff = mu1 - mu2
    # covmean = sqrtm(sigma1.dot(sigma2))
    # fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)

    print("\nmean_ssim", mean_ssim)
    print("mean_mse", mean_mse)
    # print("mean_pl", mean_pl)
    # print("fid", fid)

    return mean_ssim, mean_mse


'''
Examples:
## change the model_XXX

** For assessing synthetic images:
-- including the original frames in every prediction
./ssim.py --base results/next-image/synthetic/model_0/include_originals/

-- only predictions
./ssim.py --base results/next-image/synthetic/model_0/no_include_originals_only_predictions/


** For assessing real images:
-- including the original frames in every prediction
./ssim.py --base results/next-image/real/model_0/include_originals/

-- only predictions
./ssim.py --base results/next-image/real/model_0/no_include_originals_only_predictions/

'''
p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument('--base', type=str, default="results/next-image/real/", help='the base directory path')
p.add_argument('--source', type=str, default="original", help='the source directory name')
p.add_argument('--target', type=str, default="prediction", help='the target directory name')
p.add_argument('--gpu-id', type=int, default=0, help='the GPU device ID')

args = p.parse_args()
W.utils.set_gpu(args.gpu_id)


base_path = args.base

image_dir = f"{base_path}{args.source}/"
target_dir = f"{base_path}{args.target}/"
mean_ssim, mean_mse = calculate_metrics(image_dir, target_dir)

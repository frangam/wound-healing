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
import matplotlib.pyplot as plt


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


def calculate_iou(original, prediction, target_path):
    # # Asumir imágenes en blanco y negro con valores de 0 a 255
    # threshold = 127
    # original_bool = original > threshold
    # prediction_bool = prediction > threshold
    # Para imágenes normalizadas, ajustamos el umbral a un valor adecuado en el rango [0, 1]
    threshold = 0.5
    original_bool = original > threshold
    prediction_bool = prediction > threshold

    intersection = np.logical_and(original_bool, prediction_bool)
    union = np.logical_or(original_bool, prediction_bool)

    if np.sum(union) == 0:
        print("Advertencia: La unión de las áreas es cero, lo que resulta en una indeterminación para IoU.")
        iou_score = np.nan  # O podrías elegir devolver 0 o cualquier otro valor que maneje este caso.
    else:
        iou_score = np.sum(intersection) / np.sum(union)

    # Construcción dinámica del nuevo camino de guardado para visualización de IoU
    path_parts = target_path.split("/")
    path_parts[-2] = "IoU_metric"  # Reemplaza o inserta el nuevo nombre del directorio
    new_target_path = os.path.join(*path_parts)

    save_path = os.path.dirname(new_target_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Visualización
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(original_bool, cmap='gray')
    plt.title('Original Boolean')

    plt.subplot(1, 3, 2)
    plt.imshow(prediction_bool, cmap='gray')
    plt.title('Prediction Boolean')

    # Agregando visualización de la intersección
    plt.subplot(1, 3, 3)
    plt.imshow(intersection, cmap='gray')  # Muestra la intersección en el tercer panel
    plt.title('Intersection IoU')

    plt.savefig(new_target_path)
    plt.close()

    return iou_score



def calculate_metrics(image_dir="results/next-image/synthetic/original/", target_dir="results/next-image/synthetic/prediction/", setToWhite=True):
    image_files = sorted(os.listdir(image_dir))
    target_files = sorted(os.listdir(target_dir))
    
    pl_scores = []
    image_activations = []
    target_activations = []

    # Organizar archivos por time-step
    images_by_time_step = {}
    targets_by_time_step = {}

    for file_name in image_files:
        time_step = file_name.split('_')[-1].split('.')[0]  # Asume el formato "original_frame_wound_time-step.png"
        if time_step not in images_by_time_step:
            images_by_time_step[time_step] = []
        images_by_time_step[time_step].append(os.path.join(image_dir, file_name))

    metrics_per_time_step = {}
    for file_name in target_files:
        time_step = file_name.split('_')[-1].split('.')[0]  # Asume el formato "prediction_frame_wound_time-step.png"
        if time_step not in targets_by_time_step:
            targets_by_time_step[time_step] = []
        targets_by_time_step[time_step].append(os.path.join(target_dir, file_name))

    print("images_by_time_step", images_by_time_step)
    print("targets_by_time_step", targets_by_time_step)

    # for image_file, target_file in tqdm(zip(image_files, target_files), total=len(image_files), desc="Calculating metrics"):
    for time_step in images_by_time_step:
        ssim_scores = []
        mse_scores = []
        iou_scores = []

        for image_path, target_path in zip(images_by_time_step[time_step], targets_by_time_step.get(time_step, [])):

        # image_path = os.path.join(image_dir, image_file)
        # target_path = os.path.join(target_dir, target_file)

            print("image_path", image_path, "target_path", target_path)

            image = skimage.io.imread(image_path)
            target = skimage.io.imread(target_path)
            print(target_path)
            print("Min pixel value:", np.min(target))
            print("Max pixel value:", np.max(target))
            if setToWhite:
                threshold = 127  # This value might need adjustment
                target[target < threshold] = 0  # Black
                target[target >= threshold] = 255  # White

                # Dynamically construct the new save path
                path_parts = target_path.split("/")
                path_parts[-2] = "prediction_black_white"  # Replace or insert the new directory name
                new_target_path = os.path.join(*path_parts)

                # Ensure the new directory exists
                new_directory = os.path.dirname(new_target_path)
                if not os.path.exists(new_directory):
                    os.makedirs(new_directory)
                # Save the modified image
                print("Saving:", new_target_path)
                skimage.io.imsave(new_target_path, target)


            print(image.shape)
            print(target.shape)
            print(np.unique(image))
            print(np.unique(target))   




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
            print("ssim_score:", score)  # Print SSIM or MSE score

            # Calculate IoU
            score = calculate_iou(image, target, target_path)
            iou_scores.append(score)
            print("IoU score:", score)  # Print SSIM or MSE score


            # Calculate the MSE
            score = mse(image, target)
            mse_scores.append(score)
            print("MSE score:", score)  # Print SSIM or MSE score

        # Calcula las medias por time-step
        metrics_per_time_step[time_step] = {
            'mean_ssim': np.mean(ssim_scores),
            'mean_mse': np.mean(mse_scores),
            'mean_iou': np.mean(iou_scores)
        }


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

    # mean_ssim = sum(ssim_scores) / len(ssim_scores)
    # mean_mse = sum(mse_scores) / len(mse_scores)
    # mean_iou = sum(iou_scores) / len(iou_scores)
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
    
    # mean_ssim = np.mean(ssim_scores)
    # mean_mse = np.mean(mse_scores)
    # mean_iou = np.mean(iou_scores)
        
    # Calcula las medias globales
    global_ssim = np.mean([metrics['mean_ssim'] for metrics in metrics_per_time_step.values()])
    global_mse = np.mean([metrics['mean_mse'] for metrics in metrics_per_time_step.values()])
    global_iou = np.mean([metrics['mean_iou'] for metrics in metrics_per_time_step.values()])

    # Imprimir las métricas por cada time-step
    for time_step, metrics in metrics_per_time_step.items():
        print(f"Time-step: {time_step}")
        print(f"  Mean SSIM: {metrics['mean_ssim']:.3f}")
        print(f"  Mean MSE: {metrics['mean_mse']:.3f}")
        print(f"  Mean IoU: {metrics['mean_iou']:.3f}")
        print("")  # Añade una línea en blanco para separar los resultados por time-step

    print("\nmean_ssim", np.round(global_ssim, 3))
    print("mean_IoU", np.round(global_iou, 3))
    print("mean_mse", np.round(global_mse, 3))
    # print("mean_pl", mean_pl)
    # print("fid", fid)

    return global_ssim, global_mse, global_iou


'''
Examples:
## change the model_XXX

** For assessing synthetic images:
-- including the original frames in every prediction
./ssim.py --base results/next-image/synthetic/model_9/incremental/

./ssim.py --base results/next-image/real/model_9/incremental/


-- only predictions
./ssim.py --base results/next-image/synthetic/model_0/sequential/


** For assessing real images:
-- including the original frames in every prediction (incremental)
./ssim.py --base results/next-image/real/model_0/incremental/

-- only predictions (sequential)
./ssim.py --base results/next-image/real/model_0/sequential/

'''
p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument('-b', '--base', type=str, default="results/next-image/real/", help='the base directory path')
p.add_argument('-s', '--source', type=str, default="original", help='the source directory name')
p.add_argument('-t-', '--target', type=str, default="prediction", help='the target directory name')
p.add_argument('-g', '--gpu-id', type=int, default=1, help='the GPU device ID')

args = p.parse_args()
print("GPU ID:", args.gpu_id)
W.utils.set_gpu(args.gpu_id)


base_path = args.base

image_dir = f"{base_path}{args.source}/"
target_dir = f"{base_path}{args.target}/"
mean_ssim, mean_mse, mean_iou = calculate_metrics(image_dir, target_dir)

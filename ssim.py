import os
import skimage.io
from skimage.measure import compare_ssim

def calculate_ssim(image_dir="results/next-image/synthetic/original/", target_dir="results/next-image/synthetic/prediction/"):
    image_files = sorted(os.listdir(image_dir))
    target_files = sorted(os.listdir(target_dir))
    ssim_scores = []

    for image_file, target_file in zip(image_files, target_files):
        image_path = os.path.join(image_dir, image_file)
        target_path = os.path.join(target_dir, target_file)

        image = skimage.io.imread(image_path)
        target = skimage.io.imread(target_path)

        # Convertir las imágenes a escala de grises si es necesario
        if image.ndim > 2:
            image = skimage.color.rgb2gray(image)
        if target.ndim > 2:
            target = skimage.color.rgb2gray(target)

        # Calcular el índice SSIM
        ssim = compare_ssim(image, target)
        ssim_scores.append(ssim)

    mean_ssim = sum(ssim_scores) / len(ssim_scores)
    print("mean_ssim", mean_ssim)
    return mean_ssim


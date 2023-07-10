#!venv/bin/python3

"""
Contains functions for segment the wound in an image.

(c) All rights reserved.
original authors: Francisco M. Garcia-Moreno. 2023.

Source code:
https://github.com/frangam/wound-healing

Please see LICENSE.md for the full license document:
https://github.com/frangam/wound-healing/LICENSE.md
"""

import os
import torch
import torchvision
import pandas as pd
import scipy.stats
from scipy import stats
import mahotas.features.texture as mht




print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())


# Requirements:
# !{sys.executable} -m pip install opencv-python matplotlib > logs.log
# !{sys.executable} -m pip install 'git+https://github.com/facebookresearch/segment-anything.git' > logs.log
# !wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth > logs.log


import cv2
import matplotlib.pyplot as plt
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor



def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

# def get_mask_image(mask, random_color=False, alpha=True):
#     if random_color:
#         if alpha:
#           color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
#         else:
#           color = np.concatenate([np.random.random(3)], axis=0)
#     else:
#         if alpha:
#           color = np.array([30/255, 144/255, 255/255, 0.6])
#         else:
#           color = np.array([30/255, 144/255, 255/255])
   
#     h, w = mask.shape[-2:]

#     print("mask.shape before reshape:", mask.shape)

#     mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
#     return mask_image

def get_mask_image(mask, random_color=False, alpha=True):
    h, w = mask.shape[-2:]
    if random_color:
        # Mantener el código original para generar un color aleatorio
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        # Cambiar el color a blanco (1.0 en todos los canales)
        if alpha:
            color = np.array([1.0, 1.0, 1.0, 0.6])
        else:
            color = np.array([1.0, 1.0, 1.0])
   
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    return mask_image

def show_mask(mask, ax, random_color=False, alpha=True):
    ax.imshow(get_mask_image(mask, random_color, alpha))

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    






# Setup SAM
def setup_sam(sam_checkpoint, model_type, device):
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=200*1532 + 100)
    predictor = SamPredictor(sam)
    return mask_generator, predictor

# Predict masks
def predict_masks(predictor, image, input_point, input_label, input_boxes, multimask_output=True):
    predictor.set_image(image)
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box=input_boxes,
        multimask_output=multimask_output,
    )
    return masks, scores, logits

def get_area_perimeter(image, pixel_size=1):
    img_uint8 = image.astype(np.uint8)
    imgRGB = cv2.cvtColor(img_uint8, cv2.COLOR_BGRA2RGB)
    gray = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 100, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print("Number of contours in image:", len(contours))
    cnt = contours[0]
    area = cv2.contourArea(cnt)
    print("Area:", area)

    # Convert the area from pixels to square micrometers using the pixel size in micrometers
    area_um2 = area * pixel_size * pixel_size
    print("area_um2:", area_um2)

    # Calculate the perimeter in pixels
    perimeter_pixels = cv2.arcLength(cnt, True)
    print("perimeter_pixels:", perimeter_pixels)


    # Convert the perimeter from pixels to micrometers using the pixel size in micrometers
    perimeter_um = perimeter_pixels * pixel_size
    print("perimeter_um:", perimeter_um)


    return area_um2, perimeter_um
    

def get_area_perimeter_2(mask, image_mask_gray, pixel_size=1, wound_frame_name="monolayer_0_0"):
    area_counting_trues = np.sum(mask) * pixel_size**2
    print("recuento del area", area_counting_trues)

    img_uint8 = image_mask_gray.astype(np.uint8)
    imgRGB = cv2.cvtColor(img_uint8, cv2.COLOR_BGRA2RGB)
    gray = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2GRAY)


    # Use adaptive thresholding
    ret,thresh = cv2.threshold(gray,10,255,0)

    # Perform morphological closing
    kernel = np.ones((30,30),np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Use cv2.CHAIN_APPROX_NONE to get all the points in the contour
    # Use cv2.CHAIN_APPROX_NONE to get all the points in the contour
    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours = contours[1:]

    print("Number of contours in image:", len(contours))
    
    max_area = -1
    max_perimeter = -1
    img1 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


    # Loop through all the contours and calculate the area and perimeter for each
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        perimeter_pixels = cv2.arcLength(cnt, True)

        if area < (img1.shape[0]*img1.shape[1])-10000 and area > max_area:
            max_area = area
            max_perimeter = perimeter_pixels

            print(f"Contour {i}:")
            # print(f"Area (pixels): {area}, Max Area (pixels): {max_area}")
            # print(f"Perimeter (pixels): {perimeter_pixels}, Max Perimeter (pixels): {max_perimeter}")

            # Draw the contour on the image and annotate it with its area and perimeter
            img1 = cv2.drawContours(imgRGB, [cnt], -1, (0,255,255), 5)
            # gray_to_save = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

            os.makedirs(f"results/segments/{wound_frame_name}/", exist_ok=True)
            cv2.imwrite(f'results/segments/{wound_frame_name}/contours_{i}.png', img1)
    
    area_counting_trues = area_counting_trues / 10
    area_counting_trues_um2 = area_counting_trues # * pixel_size**2
    total_area_um2 = max_area # * pixel_size**2
    total_perimeter_um = max_perimeter * pixel_size
    print(f"Area (pixels): {area_counting_trues_um2}, Max Area (pixels): {total_area_um2}")
    print(f" Max Perimeter (pixels): {total_perimeter_um}")

    # Convert total area and perimeter from pixels to um
   

    return   area_counting_trues_um2, total_perimeter_um, total_area_um2


def calculate_glcm_features(image):
    # Calcular la matriz de coocurrencia de nivel de gris (GLCM)
    glcm = mht.haralick(image, ignore_zeros=True, return_mean=True)
    
    # Obtener las características de textura del GLCM
    contrast = glcm[2]
    correlation = glcm[4]
    energy = glcm[5]
    homogeneity = glcm[8]
    entropy = glcm[9]
    
    return contrast, correlation, energy, homogeneity, entropy




# def calculate_glcm_features(image_gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256):
#     glcm = np.zeros((levels, levels, len(distances), len(angles)))

#     for i, d in enumerate(distances):
#         for j, a in enumerate(angles):
#             shifted = np.roll(image_gray, int(np.round(d * np.cos(a))), axis=1)
#             shifted = np.roll(shifted, int(np.round(d * np.sin(a))), axis=0)
#             co_occurrence = np.zeros((levels, levels))

#             for x in range(levels):
#                 for y in range(levels):
#                     mask = (image_gray == x) & (shifted == y)
#                     co_occurrence[x, y] = np.sum(mask)

#             glcm[:, :, i, j] = co_occurrence

#     contrast = []
#     correlation = []
#     energy = []
#     homogeneity = []
#     entropy = []

#     for i in range(len(distances)):
#         for j in range(len(angles)):
#             contrast.append(stats.greycoprops(glcm[:, :, i, j], 'contrast')[0, 0])
#             correlation.append(stats.greycoprops(glcm[:, :, i, j], 'correlation')[0, 0])
#             energy.append(stats.greycoprops(glcm[:, :, i, j], 'energy')[0, 0])
#             homogeneity.append(stats.greycoprops(glcm[:, :, i, j], 'homogeneity')[0, 0])
#             entropy.append(stats.entropy(glcm[:, :, i, j].ravel()))

#     return contrast, correlation, energy, homogeneity, entropy

def calculate_texture_features(image):
    img_uint8 = image.astype(np.uint8)
    imgRGB = cv2.cvtColor(img_uint8, cv2.COLOR_BGRA2RGB)
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2GRAY)
    contrast, correlation, energy, homogeneity, entropy = calculate_glcm_features(gray)
    print("contrast", contrast,"correlation", correlation, "energy", energy, "homogeneity", homogeneity, "entropy", entropy)
    
    return  contrast, correlation, energy, homogeneity, entropy


def calculate_color_features(image):
    img_uint8 = image.astype(np.uint8)
    imgRGB = cv2.cvtColor(img_uint8, cv2.COLOR_BGRA2RGB)
    # Convertir la imagen a espacio de color HSV
    hsv = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2HSV)

    # Calcular la media y desviación estándar de los canales H, S y V
    mean_hsv = np.mean(hsv, axis=(0, 1))
    std_hsv = np.std(hsv, axis=(0, 1))

    # # Calcular el histograma de colores
    # hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    # hist = cv2.normalize(hist, hist).flatten()

    print("mean_hsv", mean_hsv, "std_hsv", std_hsv)
    mH = mean_hsv[0]
    mS = mean_hsv[1]
    mV = mean_hsv[2]
    sH = std_hsv[0]
    sS = std_hsv[1]
    sV = std_hsv[2]

    return  mH, mS, mV, sH, sS, sV

def calculate_edge_features(image):
    img_uint8 = image.astype(np.uint8)
    imgRGB = cv2.cvtColor(img_uint8, cv2.COLOR_BGRA2RGB)
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2GRAY)

    # Detectar bordes utilizando el operador de Canny
    edges = cv2.Canny(gray, 50, 150)

    # Calcular la cantidad de píxeles de borde
    edge_pixels = np.count_nonzero(edges)
    print("edge_pixels", edge_pixels)

    return edge_pixels

# def calculate_texture_features(image):
#     img_uint8 = image.astype(np.uint8)
#     imgRGB = cv2.cvtColor(img_uint8, cv2.COLOR_BGRA2RGB)
#     # Convertir la imagen a escala de grises
#     gray = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2GRAY)

#     # Calcular características de textura utilizando la matriz de co-ocurrencia de textura de niveles múltiples (MLTGLCM)
#     glcm = cv2.glcm(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
#     contrast = cv2.glcm_feature(glcm, cv2.GLCM_CONTRAST)
#     correlation = cv2.glcm_feature(glcm, cv2.GLCM_CORRELATION)
#     energy = cv2.glcm_feature(glcm, cv2.GLCM_ENERGY)
#     entropy = cv2.glcm_feature(glcm, cv2.GLCM_ENTROPY)
#     homogeneity = cv2.glcm_feature(glcm, cv2.GLCM_HOMOGENITY)
#     print("contrast", contrast, "correlation", correlation, "energy", energy, "entropy", entropy)

#     return contrast, correlation, energy, entropy, homogeneity



# Generar las heridas y segmentarlas
def generar_segmentacion_herida(heridas, sam_checkpoint, model_type, device, pixel_size=3.):
    mask_generator, predictor = setup_sam(sam_checkpoint, model_type, device)
    areas = []
    perimeters = []
    for herida in heridas:
        herida = np.array(herida)
        print(herida.shape)
        # includes = [herida.shape[1]//2, herida.shape[0]//2]
        exclude_left = [herida.shape[1]//6, herida.shape[0]//2]
        exclude_right = [herida.shape[1] - herida.shape[1]//6, herida.shape[0]//2]
        input_point = np.array([exclude_left, exclude_right])
        input_label = np.array([0, 0]) #1 include, 0 exclude
        input_boxes = np.array([herida[1]//3, 0, herida[1]-herida[1]//3, herida[0]])


        masks, scores, logits = predict_masks(predictor, herida, input_point, input_label, input_boxes)
        area_um2, perimeter_um = get_area_perimeter(herida, pixel_size=3.2)

        print("Physical area in square micrometers:", area_um2)
        print("Perimeter in micrometers:", perimeter_um)

        areas.append(area_um2)
        perimeters.append(perimeter_um)
    return areas, perimeters

"""
Contains functions for segment the wound in an image.

(c) All rights reserved.
original authors: Francisco M. Garcia-Moreno, Miguel Ángel Gutiérrez-Naranjo. 2023.

Source code:
https://github.com/frangam/wound-healing

Please see LICENSE.md for the full license document:
https://github.com/frangam/wound-healing/LICENSE.md
"""

import os
import torch
import torchvision
import pandas as pd

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

def get_mask_image(mask, random_color=False, alpha=True):
    if random_color:
        if alpha:
          color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
          color = np.concatenate([np.random.random(3)], axis=0)
    else:
        if alpha:
          color = np.array([30/255, 144/255, 255/255, 0.6])
        else:
          color = np.array([30/255, 144/255, 255/255])
   
    h, w = mask.shape[-2:]

    print("mask.shape before reshape:", mask.shape)

    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    return mask_image

def show_mask(mask, ax, random_color=False, alpha=True):
    ax.imshow(get_mask_image(mask, random_color, alpha))
    
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
    mask_generator = SamAutomaticMaskGenerator(sam)
    predictor = SamPredictor(sam)
    return mask_generator, predictor

# Predict masks
def predict_masks(predictor, image, input_point, input_label, multimask_output=False):
    predictor.set_image(image)
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=multimask_output,
    )
    return masks, scores, logits

def get_area_perimeter(image, pixel_size=3.2):
    img_uint8 = image
    imgRGB = cv2.cvtColor(img_uint8, cv2.COLOR_BGRA2RGB)
    gray = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 100, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print("Number of contours in image:", len(contours))
    cnt = contours[0]
    area = cv2.contourArea(cnt)

    # Convert the area from pixels to square micrometers using the pixel size in micrometers
    area_um2 = area * pixel_size * pixel_size

    # Calculate the perimeter in pixels
    perimeter_pixels = cv2.arcLength(cnt, True)

    # Convert the perimeter from pixels to micrometers using the pixel size in micrometers
    perimeter_um = perimeter_pixels * pixel_size

    return area_um2, perimeter_um




# Generar las heridas y segmentarlas
def generar_segmentacion_herida(heridas, sam_checkpoint, model_type, device, pixel_size=3.):
    mask_generator, predictor = setup_sam(sam_checkpoint, model_type, device)
    areas = []
    perimeters = []
    for herida in heridas:
        herida = np.array(herida)
        print(herida.shape)
        includes = [herida.shape[1]//2, herida.shape[0]//2]
        exclude_left = [herida.shape[1]//6, herida.shape[0]//2]
        exclude_right = [herida.shape[1] - herida.shape[1]//6, herida.shape[0]//2]
        input_point = np.array([includes, exclude_left, exclude_right])
        input_label = np.array([1, 0, 0]) #1 include, 0 exclude


        masks, scores, logits = predict_masks(predictor, herida, input_point, input_label)
        area_um2, perimeter_um = get_area_perimeter(herida, pixel_size=3.2)

        print("Physical area in square micrometers:", area_um2)
        print("Perimeter in micrometers:", perimeter_um)

        areas.append(area_um2)
        perimeters.append(perimeter_um)
    return areas, perimeters

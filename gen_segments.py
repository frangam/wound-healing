#!venv/bin/python3

"""
Contains functions for running the segmentation of wound images.

(c) All rights reserved.
original authors: Francisco M. Garcia-Moreno, Miguel Ángel Gutiérrez-Naranjo. 2023.

Source code:
https://github.com/frangam/wound-healing

Please see LICENSE.md for the full license document:
https://github.com/frangam/wound-healing/LICENSE.md
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm


from woundhealing.synthetic import draw_wound
from woundhealing.segment import setup_sam, predict_masks, get_area_perimeter, show_mask, show_points
from woundhealing.utils import set_gpu

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument('--gpu-id', type=int, default=1, help='the GPU device ID')
p.add_argument('--use-real', type=int, default=0, help='to use real data (1) or synthetic (0)')
p.add_argument('--pixel-size', type=float, default=3.2, help='the pixel size of your microscope camera')


args = p.parse_args()
set_gpu(args.gpu_id)

USE_SYNTHETIC = args.use_real == 0

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"

# Directorio para guardar las segmentaciones
segmentation_folder = "demo/segmentations/" if USE_SYNTHETIC else "demo/real_segmentations/"

# Leer el DataFrame desde el archivo CSV
df = pd.read_csv("demo/combined_wounds.csv") if USE_SYNTHETIC else pd.read_csv("demo/real_combined_wounds.csv")
# print(df.head())
# print(df.shape)

# setup SAM
mask_generator, predictor = setup_sam(sam_checkpoint, model_type, device)

herida_0 = df.head(1).loc[0, "WoundMatrix_0"]
herida_0 = np.array(eval(herida_0)) #convert to list

if USE_SYNTHETIC:
    herida_0 = draw_wound(herida_0) #TODO cchange for the segmented image

includes = [herida_0.shape[1]//2, herida_0.shape[0]//2]
exclude_left = [herida_0.shape[1]//6, herida_0.shape[0]//2]
exclude_right = [herida_0.shape[1] - herida_0.shape[1]//6, herida_0.shape[0]//2]
input_point = np.array([includes, exclude_left, exclude_right])
input_label = np.array([1, 0, 0]) #1 include, 0 exclude

new_ids = []
new_cell_types = []
new_times = []
new_areas = []
new_perimeters = []

# df = df.head(2) #TODO remove this line (only for tests)
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc='Segmenting...'):
    cell_type = row['CellType']
    cell_id = row['ID']
    seg_path = f"{segmentation_folder}{cell_id}/"
    os.makedirs(seg_path, exist_ok=True)

    for t, column in enumerate(df.columns):
        if column.startswith('WoundMatrix_'): #wounds in a time step t=i; WoundMatrix_0 (t=0); WoundMatrix_1 (t=1)...
            # print("row", row)
            wound_matrix = row[column] #it is a str like this: "[(250, 743, 0), (250, 742, 1), (250, 743, 2), ...]"
            if not pd.isna(wound_matrix):
               
                wound_matrix = eval(wound_matrix) #convert to list
                image_array = np.array(wound_matrix)#, dtype=np.uint8)

                if USE_SYNTHETIC:
                    image_array = draw_wound(image_array) #TODO cchange for the segmented iamge

                print("shape", image_array.shape)

                # Segmentation...
                # areas, perimeters = generar_segmentacion_herida(image_array, sam_checkpoint, model_type, device)
                masks, scores, logits = predict_masks(predictor, image_array, input_point, input_label)
                area, perimeter = get_area_perimeter(image_array, args.pixel_size)

                new_ids.append(cell_id)
                new_cell_types.append(cell_type)
                new_times.append(t)
                new_areas.append(area)  # Asegúrate de que `areas` sea un solo valor o una lista de valores del mismo tamaño que la cantidad de elementos en `wound_matrix`
                new_perimeters.append(perimeter)  # Asegúrate de que `perimeters` sea un solo valor o una lista de valores del mismo tamaño que la cantidad de elementos en `wound_matrix`

                
                cv2.imwrite(os.path.join(seg_path, f"{cell_id}_type_{cell_type}_complete_image_{t+1}.png"), image_array)
                # cv2.imwrite(os.path.join(seg_path, f"{cell_id}_type_{cell_type}_segmentation_{t+1}.png"), masks)

                plt.figure(figsize=(4,4))
                plt.imshow(image_array)
                show_mask(masks, plt.gca())
                show_points(input_point, input_label, plt.gca())
                plt.axis('off')
                plt.savefig(os.path.join(seg_path, f"{cell_id}_type_{cell_type}_segmentation_{t+1}.png"))


                # plt.figure(figsize=(4,2))
                # plt.imshow(image_array)
                # Guarda la figura en el disco
                # plt.savefig(os.path.join(seg_path, f"{cell_id}_type_{cell_type}_segmentation_{t+1}.png"))
new_df = pd.DataFrame({
    'ID': new_ids,
    'CellType': new_cell_types,
    'Time': new_times,
    'Area': new_areas,
    'Perimeter': new_perimeters
})

os.makedirs("data/", exist_ok=True)
dir = f'data/synthetic.csv' if USE_SYNTHETIC else f'data/real_synthetic.csv'
new_df.to_csv(dir, index=False)
# print(new_df)



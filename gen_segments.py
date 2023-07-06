#!venv/bin/python3

"""
Contains functions for running the segmentation of wound images.

(c) All rights reserved.
original authors: Francisco M. Garcia-Moreno. 2023.

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
import matplotlib.cm as cm
from PIL import Image
import cv2
import pickle
from tqdm import tqdm



from woundhealing.synthetic import draw_wound
from woundhealing.segment import setup_sam, predict_masks, get_area_perimeter, show_mask, show_points, show_box, get_mask_image
from woundhealing.utils import set_gpu
from woundhealing import utils


'''
original SAM: "sam/sam_vit_h_4b8939.pth"; model_type="vit_h"
MedSAM: "sam_vit_b_01ec64.pth; model_type="vit_b"

Example of run:

./gen_segments.py --gpu-id 3 --sam_checkpoint sam/medsam_tune_mask_decoder.pth --model_type vit_b
'''
p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument('--gpu-id', type=int, default=0, help='the GPU device ID')
p.add_argument('--use-real', type=int, default=1, help='to use real data (1) or synthetic (0)')
p.add_argument('--pixel-size', type=float, default=3.2, help='the pixel size of your microscope camera')
# p.add_argument('--sam_checkpoint', type=str, default="sam/medsam_tune_mask_decoder.pth", help='the Segment Anythin Model checkpoint path')
# p.add_argument('--model_type', type=str, default="vit_b", help='the Segment Anythin Model checkpoint path')


args = p.parse_args()
set_gpu(args.gpu_id)

USE_SYNTHETIC = args.use_real == 0

# sam_checkpoint = args.sam_checkpoint
# model_type = args.model_type
device = "cuda"

# Directorio para guardar las segmentaciones
segmentation_folder = "data/segmentations/" if USE_SYNTHETIC else "data/real_segmentations/"

# Leer el DataFrame desde el archivo CSV o pickle
if USE_SYNTHETIC:
    df = pd.read_csv("data/combined_synth_wounds.csv")
    herida_0 = np.array(df.head(1).loc[0, "WoundMatrix_0"])
    herida_0 = np.array(eval(herida_0)) #convert to list
    herida_0 = draw_wound(herida_0) #TODO cchange for the segmented image
    image_size = herida_0.shape
else:
    with open("data/combined_real_wounds.pkl", "rb") as file:
        df = pickle.load(file)
    image_bytes = df.loc[0, "WoundMatrix_0"]
    image_array = pickle.loads(image_bytes)
    image_size = image_array.shape[:2]
    print("image_size", image_size)

    # height, width = 1532, 2048
    # herida_0 = np.frombuffer(image_bytes, dtype=np.uint8).reshape((height, width, 3))


print("df shape", df.shape)
print("herida_0 shape", image_size)

# setup SAM
mask_generator, predictor = setup_sam("sam/sam_vit_b_01ec64.pth", "vit_b", device)
mask_generator_last_steps, predictor_last_steps = setup_sam("sam/medsam_tune_mask_decoder.pth", "vit_b", device)


# includes = [image_size[1]//2,image_size[0]//2]
# exclude_left = [image_size[1]//6, image_size[0]//2]
# exclude_right = [image_size[1] - image_size[1]//6, image_size[0]//2]
input_point = None #np.array([exclude_left, exclude_right])
input_label = None #np.array([0, 0]) #1 include, 0 exclude



new_ids = []
new_cell_types = []
new_times = []
new_areas = []
new_perimeters = []

# df = df.head(2) #TODO remove this line (only for tests)
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc='Segmenting...'):
    cell_type = row['CellType']
    print("cell_type", cell_type)
    cell_id = row['ID']
    seg_path = f"{segmentation_folder}{cell_id.split('_')[0]}/{cell_id}/"
    os.makedirs(seg_path, exist_ok=True)

    for t, column in tqdm(enumerate(df.columns), total=len(df.columns), desc='Generating masks...'):
        if column.startswith('WoundMatrix_'): #wounds in a time step t=i; WoundMatrix_0 (t=0); WoundMatrix_1 (t=1)...
            # print("row", row)
            wound_matrix = row[column] #it is a str like this: "[(250, 743, 0), (250, 742, 1), (250, 743, 2), ...]"
            if pd.isna(wound_matrix):
                break
            
            if not pd.isna(wound_matrix):
                if USE_SYNTHETIC:
                    wound_matrix = eval(wound_matrix) #convert to list
                    image_array = np.array(wound_matrix, dtype=np.uint8)
                else:    
                    image_array = pickle.loads(wound_matrix)

                # print("image_array shape", image_array.shape)

                # Segmentation...
                # areas, perimeters = generar_segmentacion_herida(image_array, sam_checkpoint, model_type, device)
                # masks = mask_generator.generate(image_array)

                if (cell_type != 1 and t+1 == utils.len_cell_type_time_step(cell_type)-1) or ((cell_type==1 and t+1 == utils.len_cell_type_time_step(cell_type)-2)):
                    input_boxes = np.array([image_size[1]//2-300, 0, image_size[1]//2+300, image_size[0]])
                    masks, scores, logits = predict_masks(predictor_last_steps, image_array, input_point, input_label, input_boxes, multimask_output=True) #multimask_output=True return 3 masks
                elif t+1 == utils.len_cell_type_time_step(cell_type) or ((cell_type==1 and t+1 == utils.len_cell_type_time_step(cell_type)-1)):
                    input_boxes = np.array([image_size[1]//2-120, 0, image_size[1]//2+120, image_size[0]])
                    masks, scores, logits = predict_masks(predictor_last_steps, image_array, input_point, input_label, input_boxes, multimask_output=True) #multimask_output=True return 3 masks
                else:
                    input_boxes = np.array([image_size[1]//3-50, 0, image_size[1]-image_size[1]//3+50, image_size[0]])
                    masks, scores, logits = predict_masks(predictor, image_array, input_point, input_label, input_boxes, multimask_output=True) #multimask_output=True return 3 masks

                # Get the indices that would sort `scores`
                sort_indices = np.argsort(scores)[::-1] #descending
                # Use these indices to sort `masks`, `scores`, and `logits`
                masks = masks[sort_indices]
                scores = scores[sort_indices]
                logits = logits[sort_indices]


                print("len mask:", len(masks))

                if len(masks) == 0:
                    break
                
                if len(masks) > 1:
                    area, perimeter = get_area_perimeter(image_array, args.pixel_size)

                    new_ids.append(cell_id)
                    new_cell_types.append(cell_type)
                    new_times.append(t)
                    new_areas.append(area)  # Asegúrate de que `areas` sea un solo valor o una lista de valores del mismo tamaño que la cantidad de elementos en `wound_matrix`
                    new_perimeters.append(perimeter)  # Asegúrate de que `perimeters` sea un solo valor o una lista de valores del mismo tamaño que la cantidad de elementos en `wound_matrix`

                    
                    #save original image
                    path1 = f"{seg_path}/original/"
                    os.makedirs(path1, exist_ok=True)
                    cv2.imwrite(os.path.join(path1, f"{cell_id}_type_{cell_type}_complete_image_{t+1}.png"), image_array)
                    # cv2.imwrite(os.path.join(seg_path, f"{cell_id}_type_{cell_type}_segmentation_{t+1}.png"), masks)


                    worse_score =+2
                    best_score =-1
                    best_mask_id = 0
                    worse_mask_id = 0
                    path1_1 = f"{seg_path}/all_masks/"
                    os.makedirs(path1_1, exist_ok=True)
                    for k, (mask, score) in enumerate(zip(masks, scores)):
                        if score > best_score:
                            best_score = score
                            best_mask_id = k
                        if score < worse_score:
                            worse_score = score
                            worse_mask_id = k
                        fig, ax = plt.subplots(figsize=(4, 4))
                        ax.imshow(image_array)                        
                        show_mask(mask, plt.gca())
                        show_box(input_boxes, ax)
                        # show_points(input_point, input_label, plt.gca())
                        plt.title(f"Mask {k+1}, Score: {score:.3f}", fontsize=18)
                        ax.axis('off')
                        plt.savefig(os.path.join(path1_1, f"{cell_id}_type_{cell_type}_segmentation_mask_{k}_{t+1}.png"), bbox_inches='tight', pad_inches=0)
                        plt.close(fig)
                    
                    print("utils.len_cell_type_time_step(cell_type)", utils.len_cell_type_time_step(cell_type))
                    #select the best fit for the mask, depending on the time in our case 
                    if t+1 == utils.len_cell_type_time_step(cell_type) or (cell_type==1 and t+1 == utils.len_cell_type_time_step(cell_type)-1):
                        print(f"[Cell type: {cell_type}]", "t:", t+1, "selecting first mask", f"[Score: {scores[-1]}]")
                        best_mask = masks[0] #first mask
                    elif (cell_type != 1 and t+1 == utils.len_cell_type_time_step(cell_type)-1) or (cell_type==1 and t+1 == utils.len_cell_type_time_step(cell_type)-2):
                        print(f"[Cell type: {cell_type}]", "t:", t+1, "selecting middle mask", f"[Score: {scores[1]}]")
                        best_mask = masks[1] #second mask
                    else:
                        print(f"[Cell type: {cell_type}]", "t:", t+1, "selecting first mask", f"[Score: {scores[0]}]")
                        best_mask = masks[0] #first mask

                    #segmentations showing points
                    path2 = f"{seg_path}/points/"
                    os.makedirs(path2, exist_ok=True)
                    fig, ax = plt.subplots(figsize=(4, 4))
                    ax.imshow(image_array)
                    show_mask(best_mask, ax)
                    show_box(input_boxes, ax)
                    # show_points(input_point, input_label, plt.gca())
                    ax.axis('off')
                    ax.set_xlim([0, image_array.shape[1]])
                    ax.set_ylim([image_array.shape[0], 0])
                    plt.savefig(os.path.join(path2, f"{cell_id}_type_{cell_type}_segmentation_points_{t+1}.png"), bbox_inches='tight', pad_inches=0)
                    plt.close(fig)

                    #RGB segmentation
                    path3 = f"{seg_path}/rgb/"
                    os.makedirs(path3, exist_ok=True)
                    fig, ax = plt.subplots(figsize=(4, 4))
                    ax.imshow(image_array)
                    show_mask(best_mask, ax)
                    ax.axis('off')
                    ax.set_xlim([0, image_array.shape[1]])
                    ax.set_ylim([image_array.shape[0], 0])
                    plt.savefig(os.path.join(path3, f"{cell_id}_type_{cell_type}_segmentation_combined_{t+1}.png"), bbox_inches='tight', pad_inches=0)
                    plt.close(fig)

                    #grayscale
                    path4 = f"{seg_path}/gray/"
                    os.makedirs(path4, exist_ok=True)
                    fig, ax = plt.subplots(figsize=(4, 4))
                    # ax.imshow(image_array, cmap='gray')  # Convertir la imagen a escala de grises
                    mask_image = get_mask_image(best_mask, plt.gca(), alpha=False)
                    mask_gray = cv2.normalize(src=mask_image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
                    img_uint8 = mask_gray.astype(np.uint8)
                    imgRGB = cv2.cvtColor(img_uint8, cv2.COLOR_BGRA2RGB)
                    mask_gray = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2GRAY)
                    # mask_gray = np.ones_like(best_mask) * 255  # Máscara completamente blanca

                    ax.imshow(mask_gray, cmap='gray')  # Convertir la imagen a escala de grises
                    ax.axis('off')
                    ax.set_xlim([0, image_array.shape[1]])
                    ax.set_ylim([image_array.shape[0], 0])
                    plt.savefig(os.path.join(path4, f"{cell_id}_type_{cell_type}_segmentation_{t+1}.png"), bbox_inches='tight', pad_inches=0)
                    plt.close(fig)


new_df = pd.DataFrame({
    'ID': new_ids,
    'CellType': new_cell_types,
    'Time': new_times,
    'Area': new_areas,
    'Perimeter': new_perimeters
})

os.makedirs("results/", exist_ok=True)
dir = f'results/synthetic_segments.csv' if USE_SYNTHETIC else f'results/real_synthetic_segments.csv'
new_df.to_csv(dir, index=False)
# print(new_df)



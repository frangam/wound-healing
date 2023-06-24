#!venv/bin/python3

"""
Contains functions for generating real wound dataset from images.

(c) All rights reserved.
original authors: Francisco M. Garcia-Moreno, Miguel Ángel Gutiérrez-Naranjo. 2023.

Source code:
https://github.com/frangam/wound-healing

Please see LICENSE.md for the full license document:
https://github.com/frangam/wound-healing/LICENSE.md
"""

import argparse
import numpy as np
import pandas as pd
import os
import cv2 as cv
import woundhealing as W
import sys
from tqdm import tqdm
import pickle



def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--gpu-id', type=int, default=0, help='the GPU device ID')
    p.add_argument('--path', type=str, default="data/", help='the directory path')
    args = p.parse_args()
    W.utils.set_gpu(args.gpu_id)

    MONOLAYER = [0, 3, 6, 9, 12, 24, 27]
    SPHERES = [0, 3, 6, 9, 12, 15]



    # Crear una lista para almacenar los DataFrames de heridas
    wound_df_list = []

    directory = f"{args.path}real_monolayer"
    print(os.listdir(directory))
    monolayer_wound_lists = [
        [
            cv.imread(os.path.join(directory, folder, f"image_{i}.tif"))
            for i in range(len(os.listdir(os.path.join(directory, folder))))
            if not str(i).startswith(".DS_Store")  # Excluir archivo .DS_Store
        ]
        for folder in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, folder))
    ]

    print(np.array(monolayer_wound_lists[0][0]).shape)
   

    directory = f"{args.path}real_spheres"
    # sphere_wound_lists = [[cv.imread(os.path.join(directory, folder, f"imagen_{i}.tiff")) for i in range(len(os.listdir(os.path.join(directory, folder))))]
    #           for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder))]


    for i in tqdm(range(len(monolayer_wound_lists))):
        monolayer_wounds = monolayer_wound_lists[i]
        monolayer_df = pd.DataFrame()
        for j, wound in tqdm(enumerate(monolayer_wounds), desc=f"Generating wounds for Monolayer {i+1}"):
            column_name = f'WoundMatrix_{j}'
            monolayer_df[column_name] = [pickle.dumps(wound)]
        
        # Asignar ID y tipo de célula en los DataFrames
        monolayer_df['ID'] = f'Monolayer_{i+1}'
        monolayer_df['CellType'] = 0
        
        # Agregar los DataFrames a la lista
        wound_df_list.append(monolayer_df)
    
    combined_df = pd.concat(wound_df_list, ignore_index=True)

    # Guardar el DataFrame combinado en un archivo pickle
    combined_df.to_pickle(f"{args.path}/combined_real_wounds.pkl")

    # Imprimir el DataFrame
    print(combined_df.head())
    print("Shape of the combined DataFrame:", combined_df.shape)

    


if __name__ == '__main__':
    main()
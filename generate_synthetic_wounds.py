#!venv/bin/python3

"""
Contains functions for generating synthetic wound creation and healing.

(c) All rights reserved.
original authors: Francisco M. Garcia-Moreno. 2023.

Source code:
https://github.com/frangam/wound-healing

Please see LICENSE.md for the full license document:
https://github.com/frangam/wound-healing/LICENSE.md
"""

import argparse
import numpy as np
import pandas as pd
import os

import woundhealing as W
from tqdm import tqdm


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--gpu-id', type=int, default=0, help='the GPU device ID')
    p.add_argument('--n', type=int, default=100, help='the number of wounds to generate')
    p.add_argument('--w', type=int, default=1024, help='the image width')
    p.add_argument('--h', type=int, default=1024, help='the image height')
    args = p.parse_args()

    W.utils.set_gpu(args.gpu_id)
    os.makedirs("demo", exist_ok=True)

    IMG_WIDTH = args.w
    IMG_HEIGHT = args.h
    
    MONOLAYER = [0, 3, 6, 9, 12, 24, 27]
    SPHERES = [0, 3, 6, 9, 12, 15, 18]

    # Pick a random start point
    seed_width  = np.rint(np.random.uniform(0, IMG_WIDTH, 1))[0].astype(np.int32)
    seed_left  = np.rint(np.random.normal(0.25 * IMG_WIDTH, 1, 1))[0].astype(np.int32)
    seed_right = np.rint(np.random.normal(0.75 * IMG_WIDTH, 1, 1))[0].astype(np.int32)

    print('Initial values:', seed_left, seed_right, seed_width)


    ## Generar múltiples heridas de tipo monolayer y sphere
    num_wounds = args.n

    monolayer_wound_lists = [W.synthetic.generate_wound(MONOLAYER, seed_left, seed_right, seed_width, IMG_WIDTH, IMG_HEIGHT) for _ in range(num_wounds)]
    sphere_wound_lists = [W.synthetic.generate_wound(SPHERES, seed_left, seed_right, seed_width, IMG_WIDTH, IMG_HEIGHT) for _ in range(num_wounds)]

    # Crear una lista para almacenar los DataFrames de heridas
    wound_df_list = []

    # Generar DataFrames para las heridas y asignar ID y tipo de célula en cada iteración
    for i in tqdm(range(len(monolayer_wound_lists)), "Generating..."):        
        monolayer_wounds = monolayer_wound_lists[i]
        # print("image shape", np.array(monolayer_wounds).shape)
        sphere_wounds = sphere_wound_lists[i]
        
        monolayer_df = W.synthetic.generate_wound_dataframe(MONOLAYER, seed_left, seed_right, seed_width, IMG_WIDTH, IMG_HEIGHT)
        sphere_df = W.synthetic.generate_wound_dataframe(SPHERES, seed_left, seed_right, seed_width, IMG_WIDTH, IMG_HEIGHT)
        
        # Asignar ID y tipo de célula en los DataFrames
        monolayer_df['ID'] = f'Monolayer_{i+1}'
        monolayer_df['CellType'] = 0
        # print(monolayer_df.head())
        # print("monolayer_df shape", monolayer_df.shape)
        
        sphere_df['ID'] = f'Sphere_{i+1}'
        sphere_df['CellType'] = 1
        
        # Agregar los DataFrames a la lista
        wound_df_list.append(monolayer_df)
        wound_df_list.append(sphere_df)

        # Generar videos y guardar los frames
        monolayer_video_name = f"data/synth_monolayer/monolayer_video_{i+1}.mp4"
        monolayer_image_folder = f"data/synth_monolayer/frames_{i+1}"
        W.synthetic.generate_video(monolayer_wounds, monolayer_video_name, monolayer_image_folder, IMG_WIDTH, IMG_HEIGHT)
        
        sphere_video_name = f"data/synth_spheres/sphere_video_{i+1}.mp4"
        sphere_image_folder = f"data/synth_spheres/frames_{i+1}"
        W.synthetic.generate_video(sphere_wounds, sphere_video_name, sphere_image_folder, IMG_WIDTH, IMG_HEIGHT)

    # Combinar los DataFrames de heridas en un solo DataFrame
    combined_df = pd.concat(wound_df_list, ignore_index=True)

    # Guardar el DataFrame combinado en un archivo CSV
    combined_df.to_csv("data/combined_synth_wounds.csv", index=False)

    print(combined_df.head())

    # Print shape of the combined DataFrame
    print("Shape of the synth_combined DataFrame:", combined_df.shape)


    # # # Generate videos
    # # generate_video(monolayer_wounds, "demo/monolayer/monolayer_video.mp4", "demo/monolayer/", IMG_WIDTH, IMG_HEIGHT)
    # # generate_video(sphere_wounds, "demo/spheres/sphere_video.mp4", "demo/spheres/", IMG_WIDTH, IMG_HEIGHT)
    # for i, row in combined_df.iterrows():
    #     print(row)
    #     monolayer_wound = row['WoundMatrix_0']  # Obtener la herida del tipo monolayer en el instante i
    #     sphere_wound = row['WoundMatrix_1']  # Obtener la herida del tipo spheres en el instante i
    #     print(monolayer_wound)

    #     # Generar el video para la herida monolayer en el instante i
    #     generate_video([monolayer_wound], f"demo/combined/monolayer_video_{i}.mp4", "demo/combined/monolayer_frames/", IMG_WIDTH, IMG_HEIGHT)

    #     # Generar el video para la herida spheres en el instante i
    #     generate_video([sphere_wound], f"demo/combined/sphere_video_{i}.mp4", "demo/combined/sphere_frames/", IMG_WIDTH, IMG_HEIGHT)


if __name__ == '__main__':
    main()

#!venv/bin/python3

"""
Contains functions for generating synthetic wound creation and healing.

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

import woundhealing as W
from tqdm import tqdm


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--gpu-id', type=int, default=0, help='the GPU device ID')
    p.add_argument('--n', type=int, default=300, help='the total number of wounds to generate')
    p.add_argument('--w', type=int, default=1024, help='the image width')
    p.add_argument('--h', type=int, default=1024, help='the image height')
    args = p.parse_args()

    W.utils.set_gpu(args.gpu_id)
    os.makedirs("demo", exist_ok=True)

    IMG_WIDTH = args.w
    IMG_HEIGHT = args.h
    
    TIME_INTERVALS = [0, 3, 6, 9, 12, 24, 27, 30]

    # Wound healing types
    healing_types = ['slow_healing', 'medium_healing', 'fast_healing']
    healing_rates = [5, 2, 1]  # Different healing rates for different types of wounds

    total_wounds = args.n
    wounds_per_type = total_wounds // len(healing_types)  # Equal division of total wounds among healing types

    wound_df_list = []
    wound_global_counter_index = 0
    for healing_type, healing_rate in zip(healing_types, healing_rates):
        for i in tqdm(range(wounds_per_type), f"Generating {healing_type} wounds..."):        
            wound_frames = W.synthetic.generate_wound(TIME_INTERVALS, IMG_WIDTH, IMG_HEIGHT, healing_rate)
            print("wound_frames shape", np.array(wound_frames).shape)

            wound_df = W.synthetic.generate_wound_dataframe(wound_frames)
            print("wound_df shape", wound_df.shape)

            wound_df['ID'] = f'{healing_type}_{i+1}'
            wound_df['HealingType'] = healing_type
            
            wound_df_list.append(wound_df)

            for i, frame in enumerate(wound_frames):
                # Create a root directory and then subdirectories for each type
                os.makedirs(f"data/synth_wounds/videos/", exist_ok=True)
                video_name = f"data/synth_wounds/videos/{healing_type}_video_{i+1}.mp4"
                image_folder = f"data/synth_wounds/"
                W.synthetic.generate_video(wound_global_counter_index, healing_type, wound_frames, video_name, image_folder, IMG_WIDTH, IMG_HEIGHT)
            wound_global_counter_index += i
            print("wound_global_counter_index", wound_global_counter_index)


    combined_df = pd.concat(wound_df_list, ignore_index=True)
    combined_df.to_csv("data/combined_synth_wounds.csv", index=False)

    print(combined_df.head())
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

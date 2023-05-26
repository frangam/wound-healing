#!venv/bin/python3
import argparse
import numpy as np
import pandas as pd
import os
from woundhealing.synthetic import generate_wound, generate_video

def generate_wound_dataframe(cell_types, seed_left, seed_right, seed_high, IMG_WIDTH=1000, IMG_HEIGHT=1000):
    """
    Generates a DataFrame with the wounds at each step of the healing process.

    Args:
        cell_types (list): The list of cell types.
        seed_left (int): Initial left edge of the wound.
        seed_right (int): Initial right edge of the wound.
        seed_high (int): Initial height of the wound.
        IMG_WIDTH (int, optional): The width of the image. Defaults to 1000.
        IMG_HEIGHT (int, optional): The height of the image. Defaults to 1000.

    Returns:
        pandas.DataFrame: A DataFrame containing the wounds at each step of the healing process.
    """
    wounds = generate_wound(cell_types, seed_left, seed_right, seed_high, IMG_WIDTH, IMG_HEIGHT)

    df = pd.DataFrame()
    for i, wound in enumerate(wounds):
        df[f'WoundMatrix_{i}'] = [wound]

    return df


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--n', type=int, default=10, help='the number of wounds to generate')
    p.add_argument('--w', type=int, default=1000, help='the image width')
    p.add_argument('--h', type=int, default=1000, help='the image height')
    args = p.parse_args()

    os.makedirs("demo", exist_ok=True)


    IMG_WIDTH = args.w
    IMG_HEIGHT = args.h
    
    MONOLAYER = [0, 3, 6, 9, 12, 24, 27]
    SPHERES = [0, 3, 6, 9, 12, 15]

    # Pick a random start point
    seed_high  = np.rint(np.random.uniform(0, IMG_HEIGHT, 1))[0].astype(np.int32)
    seed_left  = np.rint(np.random.normal(0.25 * IMG_WIDTH, 1, 1))[0].astype(np.int32)
    seed_right = np.rint(np.random.normal(0.75 * IMG_WIDTH, 1, 1))[0].astype(np.int32)

    print('Initial values:', seed_left, seed_right, seed_high)


    ## Generar múltiples heridas de tipo monolayer y sphere
    num_wounds = args.n

    monolayer_wound_lists = [generate_wound(MONOLAYER, seed_left, seed_right, seed_high, IMG_WIDTH, IMG_HEIGHT) for _ in range(num_wounds)]
    sphere_wound_lists = [generate_wound(SPHERES, seed_left, seed_right, seed_high, IMG_WIDTH, IMG_HEIGHT) for _ in range(num_wounds)]

    # Crear una lista para almacenar los DataFrames de heridas
    wound_df_list = []

    # Generar DataFrames para las heridas y asignar ID y tipo de célula en cada iteración
    for i in range(num_wounds):
        monolayer_wounds = monolayer_wound_lists[i]
        sphere_wounds = sphere_wound_lists[i]
        
        monolayer_df = generate_wound_dataframe(MONOLAYER, seed_left, seed_right, seed_high, IMG_WIDTH, IMG_HEIGHT)
        sphere_df = generate_wound_dataframe(SPHERES, seed_left, seed_right, seed_high, IMG_WIDTH, IMG_HEIGHT)
        
        # Asignar ID y tipo de célula en los DataFrames
        monolayer_df['ID'] = f'Monolayer_{i+1}'
        monolayer_df['CellType'] = 0
        
        sphere_df['ID'] = f'Sphere_{i+1}'
        sphere_df['CellType'] = 1
        
        # Agregar los DataFrames a la lista
        wound_df_list.append(monolayer_df)
        wound_df_list.append(sphere_df)

        # Generar videos y guardar los frames
        monolayer_video_name = f"demo/monolayer/monolayer_video_{i+1}.mp4"
        monolayer_image_folder = f"demo/monolayer/frames_{i+1}"
        generate_video(monolayer_wounds, monolayer_video_name, monolayer_image_folder, IMG_WIDTH, IMG_HEIGHT)
        
        sphere_video_name = f"demo/spheres/sphere_video_{i+1}.mp4"
        sphere_image_folder = f"demo/spheres/frames_{i+1}"
        generate_video(sphere_wounds, sphere_video_name, sphere_image_folder, IMG_WIDTH, IMG_HEIGHT)

    # Combinar los DataFrames de heridas en un solo DataFrame
    combined_df = pd.concat(wound_df_list, ignore_index=True)

    # Guardar el DataFrame combinado en un archivo CSV
    combined_df.to_csv("demo/combined_wounds.csv", index=False)




    # Print shape of the combined DataFrame
    print("Shape of the combined DataFrame:", combined_df.shape)


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

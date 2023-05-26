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
    p.add_argument('--n', type=int, default=5, help='the number of wounds to generate')
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

        # Generate wounds
    monolayer_wounds = generate_wound(MONOLAYER, seed_left, seed_right, seed_high, IMG_WIDTH, IMG_HEIGHT)
    sphere_wounds = generate_wound(SPHERES, seed_left, seed_right, seed_high, IMG_WIDTH, IMG_HEIGHT)

    # Generate wounds and create DataFrames
    monolayer_df = generate_wound_dataframe(MONOLAYER, seed_left, seed_right, seed_high, IMG_WIDTH, IMG_HEIGHT)
    spheres_df = generate_wound_dataframe(SPHERES, seed_left, seed_right, seed_high, IMG_WIDTH, IMG_HEIGHT)

    # Combine DataFrames into a single DataFrame
    combined_df = pd.concat([monolayer_df, spheres_df], axis=1)

    # Save combined DataFrame to CSV file
    combined_df.to_csv("demo/combined_wounds.csv", index=False)

    # Print shape of the combined DataFrame
    print("Shape of the combined DataFrame:", combined_df.shape)


    # Generate videos
    generate_video(monolayer_wounds, "demo/monolayer/monolayer_video.mp4", "demo/monolayer/", IMG_WIDTH, IMG_HEIGHT)
    generate_video(sphere_wounds, "demo/spheres/sphere_video.mp4", "demo/spheres/", IMG_WIDTH, IMG_HEIGHT)


if __name__ == '__main__':
    main()

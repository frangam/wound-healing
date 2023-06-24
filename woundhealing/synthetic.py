"""
Contains functions for simulating wound creation and healing.

(c) All rights reserved.
original authors: Francisco M. Garcia-Moreno, Miguel Ángel Gutiérrez-Naranjo. 2023.

Source code:
https://github.com/frangam/wound-healing

Please see LICENSE.md for the full license document:
https://github.com/frangam/wound-healing/LICENSE.md
"""

import os
import numpy as np
import pandas as pd
import cv2


def expand_width(left, right, IMG_WIDTH=1000):
    """
    Modifies the wound's width in each time step.

    Args:
        left (int): The left edge of the wound.
        right (int): The right edge of the wound.
        IMG_WIDTH (int, optional): The width of the image. Defaults to 1000.

    Returns:
        tuple: New positions for the left and right edges.
    """
    r1 = np.random.uniform()
    new_left = left - 1 if (r1 < 1/3 and left > 0) else left + 1 if (r1 > 2/3 and left < IMG_WIDTH - 1) else left

    r2 = np.random.uniform()
    new_right = right - 1 if (r2 < 1/3 and right > new_left + 1) else right + 1 if (r2 > 2/3 and right < IMG_WIDTH - 1) else right

    return new_left, new_right


def expand_upwards(left, right, height, IMG_WIDTH=1000, IMG_HEIGHT=1000):
    """
    Expands the wound upwards.

    Args:
        left (int): The left edge of the wound.
        right (int): The right edge of the wound.
        height (int): The current height of the wound.
        IMG_WIDTH (int, optional): The width of the image. Defaults to 1000.
        IMG_HEIGHT (int, optional): The height of the image. Defaults to 1000.

    Returns:
        list: List of tuples with the left, right, and height coordinates of the wound.
    """
    top_edge = [(left, right, height)]
    while height < IMG_HEIGHT - 1:
        left, right = expand_width(left, right, IMG_WIDTH)
        height += 1
        top_edge.append((left, right, height))
    return top_edge


def expand_downwards(left, right, height, IMG_WIDTH=1000):
    """
    Expands the wound downwards.

    Args:
        left (int): The left edge of the wound.
        right (int): The right edge of the wound.
        height (int): The current height of the wound.
        IMG_WIDTH (int, optional): The width of the image. Defaults to 1000.

    Returns:
        list: List of tuples with the left, right, and height coordinates of the wound.
    """
    bottom_edge = []
    while height > 0:
        left, right = expand_width(left, right, IMG_WIDTH)
        height -= 1
        bottom_edge.append((left, right, height))
    return list(reversed(bottom_edge))


def expand(left, right, height, IMG_WIDTH=1000, IMG_HEIGHT=1000):
    """
    Expands the wound both upwards and downwards.

    Args:
        left (int): The left edge of the wound.
        right (int): The right edge of the wound.
        height (int): The current height of the wound.
        IMG_WIDTH (int, optional): The width of the image. Defaults to 1000.
        IMG_HEIGHT (int, optional): The height of the image. Defaults to 1000.

    Returns:
        list: List of tuples with the left, right, and height coordinates of the wound.
    """
    return expand_downwards(left, right, height, IMG_WIDTH) + expand_upwards(left, right, height, IMG_WIDTH, IMG_HEIGHT)


def draw_wound(wound, IMG_WIDTH=1000, IMG_HEIGHT=1000):
    """
    Draws the wound on an image.

    Args:
        wound (list): List of tuples with the left, right, and height coordinates of the wound.
        IMG_WIDTH (int, optional): The width of the image. Defaults to 1000.
        IMG_HEIGHT (int, optional): The height of the image. Defaults to 1000.

    Returns:
        ndarray: An image of the wound.
    """
    matrix = np.zeros((IMG_WIDTH, IMG_HEIGHT))
    for l, r, h in wound:
        for k in range(l, r + 1):
            matrix[h, k] = 1
    img_rgb = np.stack((matrix,) * 3, axis=-1) 

    # Convert the matrix to uint8 (integers from 0 to 255)
    img_rgb = (img_rgb * 255).astype(np.uint8)

    return img_rgb


def generate_wound(cell_types, seed_left, seed_right, seed_high, IMG_WIDTH=1000, IMG_HEIGHT=1000, width_reduction=0.3):
    """
    Generates a wound and simulates its healing process.

    Args:
        cell_types (list): The list of cell types.
        seed_left (int): Initial left edge of the wound.
        seed_right (int): Initial right edge of the wound.
        seed_high (int): Initial height of the wound.
        IMG_WIDTH (int, optional): The width of the image. Defaults to 1000.
        IMG_HEIGHT (int, optional): The height of the image. Defaults to 1000.

    Returns:
        list: A list of wounds at each step of the healing process.
    """
    l, r = seed_left, seed_right
    wound_0 = expand(l, r, seed_high, IMG_WIDTH, IMG_HEIGHT)
    print("wound_0 shape", np.array(wound_0).shape)
    wounds = [wound_0]
    num_intervals = len(cell_types) - 1
    width_reduction_ratio = width_reduction** (1 / num_intervals)  # Ratio for reducing width in each interval

    for i in range(1, len(cell_types)):
        if l < r:
            jump = 1
            high = min(jump, max(1, r - l)) + 1
            l += np.random.randint(1, high)

        if r > l:
            jump = 1
            high = min(jump, max(1, r - l)) + 1
            r -= np.random.randint(1, high)

        if i == len(cell_types) - 1:
            l, r = (l + r) // 2, (l + r) // 2
        else:
            new_width = int((r - l) * width_reduction_ratio)
            center = (l + r) // 2
            l = center - new_width // 2
            r = center + new_width // 2

        wound_i = expand(l, r, seed_high, IMG_WIDTH, IMG_HEIGHT)
        wounds.append(wound_i)
    print("wounds shape", np.array(wounds).shape)

    return wounds







def generate_video(wounds, video_name, image_folder, IMG_WIDTH=1000, IMG_HEIGHT=1000):
    """
    Generates a video of the wound healing process and save the sequence of images.

    Args:
        wounds (list): A list of wounds at each step of the healing process.
        video_name (str): The name of the output video file.
        image_folder (str): The path to the folder where the sequence of images will be saved.
        IMG_WIDTH (int, optional): The width of the image. Defaults to 1000.
        IMG_HEIGHT (int, optional): The height of the image. Defaults to 1000.
    """
    os.makedirs(image_folder, exist_ok=True)

    # Check and create the directory for the video file if it doesn't exist
    video_directory = os.path.dirname(video_name)
    if video_directory and not os.path.exists(video_directory):
        os.makedirs(video_directory)


    # Create the VideoWriter object
    video_output = video_name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_output, fourcc, 10.0, (IMG_WIDTH, IMG_HEIGHT))

    # Generate the video and save images
    for i, wound in enumerate(wounds):
        img_rgb = draw_wound(wound, IMG_WIDTH, IMG_HEIGHT)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        video.write(img_bgr)

        # Save the frame as an image file
        cv2.imwrite(os.path.join(image_folder, f'image_{i}.png'), img_bgr)

    # Release the resources
    video.release()



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
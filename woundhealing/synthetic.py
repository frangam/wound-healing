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


def expand_rightwards(left, right, width, IMG_WIDTH=1000):
    """
    Expands the wound upwards.

    Args:
        left (int): The left edge of the wound.
        right (int): The right edge of the wound.
        width (int): The current width of the wound.
        IMG_WIDTH (int, optional): The width of the image. Defaults to 1000.

    Returns:
        list: List of tuples with the left, right, and height coordinates of the wound.
    """
    top_edge = [(left, right, width, False)] #False: indicates no cut in height
    while width < IMG_WIDTH - 1:
        left, right = expand_width(left, right, IMG_WIDTH)
        width += 1
        top_edge.append((left, right, width, False)) #False: indicates no cut in height
    return top_edge


def expand_leftwards(left, right, width, IMG_WIDTH=1000):
    """
    Expands the wound downwards.

    Args:
        left (int): The left edge of the wound.
        right (int): The right edge of the wound.
        width (int): The current width of the wound.
        IMG_WIDTH (int, optional): The width of the image. Defaults to 1000.

    Returns:
        list: List of tuples with the left, right, and height coordinates of the wound.
    """
    bottom_edge = []
    while width > 0:
        left, right = expand_width(left, right, IMG_WIDTH)
        width -= 1
        bottom_edge.append((left, right, width, False)) #False: indicates no cut in height
    return list(reversed(bottom_edge))


def expand_height(top, bottom, height, IMG_HEIGHT=1000):
    """
    Modifies the wound's height in each time step.

    Args:
        top (int): The top edge of the wound.
        bottom (int): The bottom edge of the wound.
        IMG_HEIGHT (int, optional): The height of the image. Defaults to 1000.

    Returns:
        tuple: New positions for the top and bottom edges.
    """
    r = np.random.uniform()
    new_top = top - 1 if (r < 1/3 and top > 0) else top + 1 if (r > 2/3 and top < IMG_HEIGHT - 1) else top

    r = np.random.uniform()
    new_bottom = bottom - 1 if (r < 1/3 and bottom > new_top + 1) else bottom + 1 if (r > 2/3 and bottom < IMG_HEIGHT - 1) else bottom

    return new_top, new_bottom


def expand_upwards(top, bottom, height, IMG_HEIGHT=1000):
    """
    Expands the wound upwards.

    Args:
        top (int): The top edge of the wound.
        bottom (int): The bottom edge of the wound.
        height (int): The current height of the wound.
        IMG_HEIGHT (int, optional): The height of the image. Defaults to 1000.

    Returns:
        list: List of tuples with the top, bottom, and height coordinates of the wound.
    """
    wound = [(top, bottom, height)]
    while height < IMG_HEIGHT - 1:
        top, bottom = expand_height(top, bottom, IMG_HEIGHT)
        height += 1
        wound.append((top, bottom, height))
    return wound


def expand_downwards(top, bottom, height, IMG_HEIGHT=1000):
    """
    Expands the wound downwards.

    Args:
        top (int): The top edge of the wound.
        bottom (int): The bottom edge of the wound.
        height (int): The current height of the wound.
        IMG_HEIGHT (int, optional): The height of the image. Defaults to 1000.

    Returns:
        list: List of tuples with the top, bottom, and height coordinates of the wound.
    """
    wound = []
    while height > 0:
        top, bottom = expand_height(top, bottom, IMG_HEIGHT)
        height -= 1
        wound.append((top, bottom, height))
    return list(reversed(wound))


def expand(left, right, width, IMG_WIDTH=1000):
    """
    Expands the wound in both width and height.

    Args:
        left (int): The left edge of the wound.
        right (int): The right edge of the wound.
        width (int): The current width of the wound.
        IMG_WIDTH (int, optional): The width of the image. Defaults to 1000.

    Returns:
        list: List of tuples with the left, right, and height coordinates of the wound.
    """
    return (
        expand_leftwards(left, right, width, IMG_WIDTH)
        + expand_rightwards(left, right, width, IMG_WIDTH)
    )

def generate_wound(time_intervals, seed_left, seed_right, seed_width, IMG_WIDTH=1000, IMG_HEIGHT=1000, width_reduction=0.08, num_height_intervals=2, height_cut_max_size=100):
    """
    Generates a wound and simulates its healing process.

    Args:
        time_intervals (list): The list of time intervals in hours.
        seed_left (int): Initial left edge of the wound.
        seed_right (int): Initial right edge of the wound.
        seed_width (int): Initial width of the wound.
        IMG_WIDTH (int, optional): The width of the image. Defaults to 1000.
        IMG_HEIGHT (int, optional): The height of the image. Defaults to 1000.
        width_reduction (float, optional): Width reduction ratio for each interval. Defaults to 0.08.
        num_height_intervals (int, optional): Number of intervals in which height reduction will occur from the last intervals.
                                              Defaults to 2.

    Returns:
        list: A list of wounds at each step of the healing process.
    """
    l, r = seed_left, seed_right
    seed_width = seed_right - seed_left
    wound_0 = expand(l, r, seed_width, IMG_WIDTH)
    wounds = [wound_0]
    num_intervals = len(time_intervals) - 1
    width_reduction_ratio = width_reduction ** (1 / num_intervals)  # Ratio for reducing width in each interval

    height_cuts = []  # List to store the height cuts

    for i in range(1, len(time_intervals)):
        if l < r:
            jump = 1
            width = min(jump, max(1, r - l)) + 1
            l += np.random.randint(1, width)

        if r > l:
            jump = 1
            width = min(jump, max(1, r - l)) + 1
            r -= np.random.randint(1, width)

        if i == len(time_intervals) - 1:
            new_width = int((r - l) * width_reduction)
            center = (l + r) // 2
            l = center - new_width // 2
            r = center + new_width // 2
        else:
            new_width = int((r - l) * width_reduction_ratio)
            center = (l + r) // 2
            l = center - new_width // 2
            r = center + new_width // 2
        
        wound_i = expand(l, r, seed_width, IMG_WIDTH)
        wounds.append(wound_i)

    # mark vertical holes
    wounds_res = []
    cuts = []  # Here we will store the cuts

    for i, wound in enumerate(wounds):
        # Check if height reduction should occur in the last intervals
        if i >= num_intervals - num_height_intervals-1:
            # Add new cuts
            num_cuts = np.random.randint(0, 4)  # Random number of height cuts between 0 and 3
            if num_cuts > 0:
                cut_heights = np.random.randint(100, 300, size=num_cuts)  # Random size 
                priority_zones_of_cut = [0, IMG_HEIGHT//2, IMG_HEIGHT]
                
                for ch in cut_heights:
                    select_coord_cuts = np.random.randint(0, len(priority_zones_of_cut))  
                    coord_start = priority_zones_of_cut[select_coord_cuts]                
                    new_cut = [coord_start, coord_start+ch]  # Cut is the height coord and the hegiht of the cut
                    # Add new cut to the existing ones
                    cuts.append(new_cut)
                    
            # Apply all the cuts
            for cut_start_height, cut_size in cuts:
                for k in range(len(wound)):
                    if len(wound[k]) == 4:
                        l, r, h, remove = wound[k]
                        if h >= cut_start_height and h <= cut_size:
                            wound[k] = (l, r, h, True)
                                
        wounds_res.append(wound)


    return wounds_res












def draw_wound(wound, IMG_WIDTH=1000, IMG_HEIGHT=1000):
    """
    Draws the wound on an image.

    Args:
        wound (list): List of tuples with the left, right, height, and remove_flag coordinates of the wound.
        IMG_WIDTH (int, optional): The width of the image. Defaults to 1000.
        IMG_HEIGHT (int, optional): The height of the image. Defaults to 1000.

    Returns:
        ndarray: An image of the wound.
    """
    matrix = np.zeros((IMG_WIDTH, IMG_HEIGHT))
    for value in wound:

        l, r, h, remove_flag = value if len(value) == 4 else (*value, False)
        if not remove_flag:
            matrix[h, l:r+1] = 1
        # else:
        #     print("no draw")

    img_rgb = np.repeat(matrix[:, :, np.newaxis], 3, axis=2)
    img_rgb = (img_rgb * 255).astype(np.uint8)

    return img_rgb



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

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

    # control the closure of wound when left edge overpass the right edge
    print("new_left", new_left)
    if new_left > new_right:
        print("new_left >new_right ")

        new_left = new_right

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

def generate_wound(time_intervals, IMG_WIDTH, IMG_HEIGHT, healing_rate=1.0, width_reduction_range=(0.1, 0.3), larger_width_reduction=0.9, num_height_intervals=2):
    # Pick a random start point for the wound
    seed_width = np.rint(np.random.uniform(IMG_WIDTH//3, IMG_WIDTH, 1))[0].astype(np.int32)
    seed_left = np.rint(np.random.normal(0.25 * IMG_WIDTH, 1, 1))[0].astype(np.int32)
    seed_right = np.rint(np.random.normal(0.75 * IMG_WIDTH, 1, 1))[0].astype(np.int32)

    print('Initial values:', seed_left, seed_right, seed_width)

    wound_frames = []
    closure_interval = None

    for i, time in enumerate(time_intervals):

        # Expand the wound based on the current width
        frame = expand(seed_left, seed_right, seed_width, IMG_WIDTH)

        # Calculate width reduction
        width_reduction_ratio = np.random.uniform(*width_reduction_range) * healing_rate
        new_width = int(seed_width * width_reduction_ratio)

        # If the wound width is below a certain threshold, close the wound directly
        if new_width < 30:  # Change this threshold based on your needs
            new_width = 0

        # Update the wound dimensions for the next frame
        seed_left = seed_left + seed_width // 2 - new_width // 2
        seed_right = seed_left + new_width
        seed_width = new_width

        # Estimate the closure interval if it hasn't been estimated yet
        if closure_interval is None:
            closure_interval = estimate_closure_interval(wound_frames, time_intervals[i+1:])

        # Apply the logic of cutting the wound in the closure interval and subsequent frames
        if closure_interval is not None and i >= closure_interval:
            frame = cut_wound(frame, closure_interval, num_height_intervals, IMG_HEIGHT)

        wound_frames.append(frame)

    return wound_frames




def wound_closure_sum_diff(frames):
    initial_frame_width = calculate_total_width(frames[0])
    print("initial_frame", np.array(frames[0]).shape, "width:", initial_frame_width)

    current_frame_width = calculate_total_width(frames[0])
    print("current_frame", np.array(frames[-1]).shape, "width:", current_frame_width)

    return initial_frame_width - current_frame_width

def calculate_wound_closure_ratio(frames):
    """
    Calculate the ratio of wound closure based on the frames.

    Args:
        frames (list): List of tuples representing the wound frames.

    Returns:
        float: Ratio of wound closure between 0 and 1.
    """
    if len(frames) <= 1:
        return 0.0

    initial_frame_width = calculate_total_width(frames[0])
    current_frame_width = calculate_total_width(frames[-1])
    closure_ratio = (initial_frame_width - current_frame_width) / initial_frame_width

    print("closure_ratio", closure_ratio)

    return closure_ratio



def calculate_total_width(frame):
    """
    Calculate the total width given a frame.

    Args:
        frame (list): List of tuples representing a single frame of the wound.
        the frame shape is (number of coord, 4) :  4 >>>(lefth edge; right, height, hidde this coord in graphic) 
        Example: (1024, 4) >>> 
        position 0 - (230, 567, 0, False)
        position 1 - (241, 579, 1, True)
        ..
        position 1023 - (215, 598, 1023, False)

    Returns:
        int: Total width of the wound in the frame.
    """
    total_width_sum = 0
    # print("frame", np.array(frame).shape)

    for height in frame:
        left, right, _, _ = height
        total_width_sum += right - left

    return total_width_sum





def cut_wound(frame, closure_interval, num_height_intervals, IMG_HEIGHT):
    """
    Applies the logic of cutting the wound in the given frame.

    Args:
        frame (tuple): The frame of the wound to be cut.
        closure_interval (int): The remaining closure interval for the wound.
        num_height_intervals (int): The number of height intervals for cutting.
        IMG_HEIGHT (int): The height of the image.

    Returns:
        tuple: Updated frame with cuts if applicable.
    """
    print("frame to cut", np.array(frame).shape)
    print("closure_interval", closure_interval)
    for k in range(len(frame)):
        l, r, h, *remove = frame[k]
        frame[k] = (l, r, h, False)


    # Check if the frame meets the conditions for cutting
    if closure_interval <= num_height_intervals:
        wound_closure_progress = (closure_interval - 1) / num_height_intervals

        # Check if it is the last frame for wound closure
        if closure_interval == 1:
            # Cut from the top and bottom with larger cut size
            cut_start_height = 0
            cut_size = IMG_HEIGHT
            for k in range(len(frame)): #iterate over all the coords
                l, r, h, *remove = frame[k]
                if h >= cut_start_height and h <= cut_size:
                    frame[k] = (l, r, h, True)
        else:
            # Choose random cutting zones
            priority_zones_of_cut = [0, IMG_HEIGHT // 2, IMG_HEIGHT]
            num_cuts = max(0, int(np.round(wound_closure_progress * 4)))  # Adjust the scaling factor as needed

            for _ in range(num_cuts):
                select_coord_cuts = np.random.randint(0, len(priority_zones_of_cut))
                coord_start = priority_zones_of_cut[select_coord_cuts]
                cut_start_height = np.random.randint(coord_start, IMG_HEIGHT)
                cut_size = np.random.randint(cut_start_height, IMG_HEIGHT)

                for k in range(len(frame)): #iterate over all the coords
                    l, r, h, *remove = frame[k]
                    if h >= cut_start_height and h <= cut_size:
                        frame[k] = (l, r, h, True)

    return frame





def estimate_closure_interval(frames, remaining_intervals, num_height_intervals=2):
    """
    Estimates the closure interval in which the wound is predicted to close based on the current frames and remaining intervals.

    Args:
        frames (list): List of frames representing the wound.
        remaining_intervals (list): List of remaining time intervals.
        num_height_intervals (int): Number of intervals to consider for closure estimation.

    Returns:
        int: Estimated closure interval or None if it cannot be estimated.
    """
    closure_ratio = calculate_wound_closure_ratio(frames)
    remaining_frames = len(remaining_intervals)

    if remaining_frames <= num_height_intervals:
        return None

    for i in range(1, num_height_intervals + 1):
        estimated_closure_progress = i / num_height_intervals
        print("estimated_closure_progress i/num_height_intervals", "i=", i, "num_height_intervals:", num_height_intervals)
        print("closure_ratio >= estimated_closure_progress", closure_ratio >= estimated_closure_progress)
        if closure_ratio >= estimated_closure_progress:
            return len(remaining_intervals) - remaining_frames + i

    return None












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
        # print("values inside the wound", np.array(value).shape)
        l, r, h, remove_flag = value if len(value) == 4 else (*value, False)
        if not remove_flag:
            matrix[h, l:r+1] = 1
            # else:
            #     print("no draw")

    img_rgb = np.repeat(matrix[:, :, np.newaxis], 3, axis=2)
    img_rgb = (img_rgb * 255).astype(np.uint8)

    return img_rgb




def generate_video(wound_global_counter_index, healing_type, wound, video_name, image_folder, IMG_WIDTH=1000, IMG_HEIGHT=1000):
    """
    Generates a video of the wound healing process and save the sequence of images.

    Args:
        wound (list): A list of frames in a  wound at each step of the healing process.
        video_name (str): The name of the output video file.
        image_folder (str): The path to the folder where the sequence of images will be saved.
        IMG_WIDTH (int, optional): The width of the image. Defaults to 1000.
        IMG_HEIGHT (int, optional): The height of the image. Defaults to 1000.
    """

    # Check and create the directory for the video file if it doesn't exist
    video_directory = os.path.dirname(video_name)
    if video_directory and not os.path.exists(video_directory):
        os.makedirs(video_directory)

    # Create the VideoWriter object
    video_output = video_name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_output, fourcc, 10.0, (IMG_WIDTH, IMG_HEIGHT))

    # Generate the video and save images
    for i, frame in enumerate(wound):
        path = f"{image_folder}wound_{wound_global_counter_index}/"
        os.makedirs(path, exist_ok=True)

        # print("frame shape", np.array(frame).shape)
        img_rgb = draw_wound(frame, IMG_WIDTH, IMG_HEIGHT)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        video.write(img_bgr)

        # Save the frame as an image file
        cv2.imwrite(os.path.join(path, f'{healing_type}_image_{i}.png'), img_bgr)

    # Release the resources
    video.release()


def generate_wound_dataframe(wounds):
    
    # wounds = generate_wound(cell_types, seed_left, seed_right, seed_high, IMG_WIDTH, IMG_HEIGHT)

    df = pd.DataFrame()
    for i, wound in enumerate(wounds):
        df[f'WoundMatrix_{i}'] = [wound]

    return df

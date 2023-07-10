"""
Contains functions for loading the wound dataset.

(c) All rights reserved.
original authors: Francisco M. Garcia-Moreno. 2023.

Source code:
https://github.com/frangam/wound-healing

Please see LICENSE.md for the full license document:
https://github.com/frangam/wound-healing/LICENSE.md
"""


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from PIL import Image
from . import utils

def load_images(base_dir='data/', image_type='synth_monolayer', remove_first_frame=False, resize_width=None, resize_height=None, remove_types=['synth_monolayer', 'real_monolayer', 'synth_spheres', 'real_spheres'], load_max=100, fill=False):
    """
    Load a dataset of images from specified directories.

    This function reads image data from directories of a specified type,
    within a base directory. It can also optionally remove the first frame
    from the 'monolayer' type images. Images are returned as a numpy array.

    Parameters:
    base_dir (str): The base directory from which to read image data.
                    Defaults to 'data/'.
    image_type (str): The type of images to load. This should correspond
                      to a subdirectory within the base directory.
                      Choices are 'synth_monolayer', 'synth_spheres', 'real_monolayer', 'real_spheres'.
                      Defaults to 'synth_monolayer'.
    remove_first_frame (bool): Whether to remove the first frame from the
                               'monolayer' images. Defaults to False.
    resize_width (int): The desired width for resizing the images.
                        Defaults to None (no resizing).
    resize_height (int): The desired height for resizing the images.
                         Defaults to None (no resizing).
    remove_types (list): A list of image types that should have the first frame removed.
                            Defaults to ['synth_monolayer', 'real_monolayer'].

    Returns:
    np.array: A numpy array of the loaded and resized images.
    """
    all_images = []
    # frame_dirs = [d for d in os.listdir(os.path.join(base_dir, image_type)) if 'frames_' in d]
    # frame_dirs.sort(key=lambda x: int(x.split('_')[1])) # Sort directories in order

    initial_frame_path = os.path.join(base_dir, image_type)
    if image_type.startswith('aug') or image_type.startswith('synth'):
        frame_dirs = [d for d in os.listdir(initial_frame_path) if 'frames_' in d]
        frame_dirs.sort(key=lambda x: int(x.split('_')[1]))  # Sort directories in order
    elif image_type.startswith('real_monolayer'):
        initial_frame_path = os.path.join(base_dir, "real_segmentations/Monolayer/")
        frame_dirs = [d for d in os.listdir(initial_frame_path) if d.startswith('Monolayer_')]
        frame_dirs.sort(key=lambda x: int(x.split('_')[1][1:]) if x.split('_')[1][1:].isdigit() else float('inf'))
    elif image_type.startswith('real_spheres'):
        initial_frame_path = os.path.join(base_dir, "real_segmentations/Sphere/")
        frame_dirs = [d for d in os.listdir(initial_frame_path) if d.startswith('Sphere_')]
        frame_dirs.sort()  # Sort directories in order

    if load_max > 0:
        frame_dirs = frame_dirs[:load_max]
    
    print("initial_frame_path", initial_frame_path)
    print("frame_dirs", frame_dirs)

    for dir in tqdm(frame_dirs, desc='Loading frames'):
        frame_images = []  # List to hold images of a single frame sequence
        if image_type.startswith('real'):
            frame_path = os.path.join(initial_frame_path, dir, "gray")
        else:
            frame_path = os.path.join(initial_frame_path, dir)

        frame_files = os.listdir(frame_path)

        # print(frame_files)
        frame_files.sort(key=lambda x: int(x.split('.')[0].split('_')[1])) # Sort frame files in order
        
        if remove_first_frame and image_type in remove_types:
            frame_files = frame_files[1:] # Remove the first frame if the condition is met
        len_f = len(frame_files)
        for t, file in enumerate(tqdm(frame_files, desc='Loading images', leave=False)):
            image = cv2.imread(os.path.join(frame_path, file))
            # Resize the image if desired
            if resize_width and resize_height:
                image = cv2.resize(image, (resize_width, resize_height))
            

            # # --- add the last image to fill no photos taken
            if fill and (('real_monolayer' in image_type and t+1 == utils.len_cell_type_time_step(0)-2) or ('spheres' in image_type and t+1 == utils.len_cell_type_time_step(1))):
                if 'real_monolayer' in image_type:
                    frame_images.append(image)
                    
                    int_no_photos = utils.intervals_no_hours_taken_photo(0)
                    print("int_no_photos", int_no_photos)
                    # print("image shape", image.shape)
                    aug = augment_frames([image], 1, r=0)
                    aug = np.squeeze(aug)
                    frame_images.append(aug)
                    for i in range(int_no_photos-1): #we need 1 additional
                        aug = augment_frames([aug], 1, r=i+1)
                        aug = np.squeeze(aug)
                        frame_images.append(aug)
                elif 'spheres' in image_type:
                    black_frame = np.zeros_like(image)
                    if image_type == "real_spheres": #due to the segmentations we repeated the last segmentation... now we remove it
                        # frame_images = frame_images[:-1]
                        frame_images.append(black_frame)

                    frame_images.append(black_frame)
                    int_no_photos = utils.intervals_no_hours_taken_photo(1)
                    for _ in range(int_no_photos+1): #we need 1 additional
                        frame_images.append(black_frame)
            else:
                frame_images.append(image)
                        
                    

        # if 'spheres' in image_type:
        #     black_frame = np.zeros_like(image)  # Crear una imagen negra del mismo tamaño
        #     # if image_type == "real_spheres":
        #     #     frame_images = frame_images[:-1]
        #     frame_images.append(black_frame) #adding black frame at the end of the sequence

        all_images.append(frame_images)
    
    
    dataset = np.array(all_images)

    print("dataset shape", dataset.shape)
    # Add a channel dimension since the images are grayscale.
    # dataset = np.expand_dims(dataset, axis=-1)  

    return dataset


def split_dataset(dataset, labels, test_ratio=0.9, seed=43):
    """
    Split dataset into training and validation sets.

    Parameters:
    dataset (np.array): The full dataset to be split.
    test_ratio (float): The ratio of the dataset to be used for testing. Defaults to 0.9.

    Returns:
    Tuple[np.array, np.array]: The training and validation datasets.
    """
    np.random.seed(seed)
    # indexes = np.arange(dataset.shape[0])
    # np.random.shuffle(indexes)
    # train_index = indexes[: int(train_ratio * dataset.shape[0])]
    # val_index = indexes[int(train_ratio * dataset.shape[0]) :]
    # train_dataset = dataset[train_index]
    # val_dataset = dataset[val_index]
    from sklearn.model_selection import train_test_split
    print(labels)
    if labels is not None and len(labels) > 0:
        print("Proporción en el conjunto de inicial:")
        print(np.bincount(labels) / len(labels))
        print("test_size",test_ratio)

        train_dataset, val_dataset, train_labels, val_labels = train_test_split(dataset, labels, test_size=test_ratio, stratify=labels, random_state=seed)
    
        print("Proporción en el conjunto de entrenamiento:")
        print(np.bincount(train_labels) / len(train_labels))

        print("Proporción en el conjunto de validación:")
        print(np.bincount(val_labels) / len(val_labels))

    else:
        train_dataset, val_dataset = train_test_split(dataset, test_size=test_ratio, random_state=seed)


    return train_dataset, val_dataset

def normalize_data(dataset):
    """
    Normalize dataset to 0-1 range.

    Parameters:
    dataset (np.array): The dataset to be normalized.

    Returns:
    np.array: The normalized dataset.
    """
    print(dataset.shape)
    return dataset / 255

def create_shifted_frames(data):
    """
    Create shifted frames for model training.

    Parameters:
    data (np.array): The dataset used to create shifted frames.

    Returns:
    Tuple[np.array, np.array]: The input data and the shifted frames.
    """
    x = data[:, 0 : data.shape[1] - 1, :, :]
    y = data[:, 1 : data.shape[1], :, :]
    return x, y

def visualize_example(data, example_index, save_path):
    """
    Visualize example frames in the dataset and save to an image file.

    Parameters:
    data (np.array): The dataset to visualize from.
    example_index (int): The index of the example to visualize.
    save_path (str): The path to save the visualization.
    """
    num_examples = data.shape[0]
    num_frames = data.shape[1]
    sqrt_num_frames = int(np.ceil(np.sqrt(num_frames)))

    if example_index >= num_examples:
        print(f"Invalid example index. Available examples: {num_examples}")
        return

    fig, axes = plt.subplots(sqrt_num_frames, sqrt_num_frames, figsize=(10, 8))

    for idx, ax in enumerate(axes.flat):
        if idx < num_frames:
            ax.imshow(data[example_index][idx], cmap="gray")
            ax.set_title(f"Frame {idx + 1}")
        ax.axis("off")

    print(f"Displaying frames for example {example_index}.")

    # Save the figure as an image
    plt.savefig(save_path)
    plt.close(fig)




def frames_to_video(data, example_index, save_path):
    """
    Save example frames in the dataset to a video file.

    Parameters:
    data (np.array): The dataset to create a video from.
    example_index (int): The index of the example to create a video from.
    save_path (str): The path to save the video.
    """
    height, width = data.shape[2], data.shape[3]
    video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 1, (width, height))

    for frame in data[example_index]:
        video.write(cv2.cvtColor(np.uint8(frame * 255), cv2.COLOR_RGB2BGR))
    video.release()


from skimage import color
def augment_frames(sequences, n, r=0, SEED=33):
    import imgaug as ia
    import imgaug.augmenters as iaa
# 
    red_width = iaa.Affine(scale={"x": (0.95, 1.0), "y": 1})
    # red_width = iaa.PiecewiseAffine(scale=(0.01, 0.05))

    if r == 1:
        red_width = iaa.Affine(scale={"x": (0.8, .9), "y": 1})
        # red_width = iaa.PiecewiseAffine(scale=(0.01, 0.08))

    elif r>1:
        # red_width = iaa.PiecewiseAffine(scale=(0.01, 0.15))
        red_width = iaa.Affine(scale={"x": (0.75, .85), "y": 1})

    seq = iaa.Sequential([
    # iaa.Fliplr(0.5), # voltea 50% de las imágenes de forma horizontal
    iaa.Affine(rotate=(-0.1, .1)) , # Rotar entre -25 y 25 grados
    # iaa.PiecewiseAffine(scale=(0.01, 0.05)) , # Transformación elástica: Útil para imágenes donde objetos se deforman, como imágenes biomédicas, 
    red_width

], random_order=True)
    random_state = ia.new_random_state(SEED)
    sequences_augmented = [seq(images=images_seq) for _ in range(n) for images_seq in sequences]
    sequences_augmented = np.array(sequences_augmented)
    # sequences_augmented = (sequences_augmented * 255).astype(np.uint8)


    print("sequences_augmented", sequences_augmented.shape)
    return sequences_augmented


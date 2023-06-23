#!venv/bin/python3

"""
Contains functions for building a model to predict the next whound image.

(c) All rights reserved.
original authors: Francisco M. Garcia-Moreno, Miguel Ángel Gutiérrez-Naranjo. 2023.

Source code:
https://github.com/frangam/wound-healing

Please see LICENSE.md for the full license document:
https://github.com/frangam/wound-healing/LICENSE.md
"""

import os
import argparse
import numpy as np

from matplotlib.animation import FuncAnimation
import io
from PIL import Image
from IPython.display import display
import cv2
import imageio


import matplotlib.pyplot as plt
from tensorflow import keras

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model


import woundhealing as W

def predict_and_visualize(model, val_dataset, save_path, num_frames_to_show=6, example_index=0):
    """
    Generate and visualize predicted frames.

    Parameters:
    - model (keras.Model): The trained model.
    - val_dataset (np.array): The validation dataset.
    - save_path (str): The path to save the visualization.
    - num_frames_to_show (int): The number of frames to show in the visualization. 
      Defaults to 6.
    - example_index (int): The index of the example from the validation dataset to visualize. 
      Defaults to 0.
    """
    example = val_dataset[example_index]
    # frames = example[:num_frames_to_show, ...]
    # original_frames = example[-num_frames_to_show:, ...]
    frames = example[:, ...]
    original_frames = example[:, ...]
    new_predictions = np.zeros(shape=(num_frames_to_show, *frames[0].shape))

    for i in range(num_frames_to_show):
        # Extract the model's prediction and post-process it.
        frames = example[: i + 1, ...]
        new_prediction = model.predict(np.expand_dims(frames, axis=0))
        print("new_prediction", new_prediction.shape)
        new_prediction = np.squeeze(new_prediction, axis=0)
        predicted_frame = np.expand_dims(new_prediction[-1, ...], axis=0)
        print("predicted_frame",  predicted_frame.shape)
        # frames = np.concatenate((frames, predicted_frame), axis=0)

        predicted_frame = np.squeeze(predicted_frame)
        predicted_frame = predicted_frame * 255  # No es necesario convertir a uint8 en este punto
        predicted_frame = predicted_frame.astype(np.uint8)
        print("predicted_frame",  predicted_frame.shape)

    
        image_filename = os.path.join(save_path, f"prediction_frame_{example_index}_{i}.png")
        plt.imsave(image_filename, predicted_frame)

        image_filename = os.path.join(save_path, f"original_frame_{example_index}_{i}.png")
        plt.imsave(image_filename, frames[i])

    fig, axes = plt.subplots(2, num_frames_to_show, figsize=(num_frames_to_show*2, 4))

    for idx, ax in enumerate(axes[0]):
        ax.imshow(np.squeeze(original_frames[idx]), cmap="gray")
        ax.set_title(f"Original Frame {idx + 1}")
        ax.axis("off")

    # new_frames = frames[num_frames_to_show:, ...]
    for idx, ax in enumerate(axes[1]):
        ax.imshow(np.squeeze(new_prediction[idx]), cmap="gray")
        ax.set_title(f"Predicted Frame {idx + 1}")
        ax.axis("off")

    plt.savefig(f"{save_path}predictions_{example_index}")
    plt.show()

def save_gif(frames, filename, duration=100):
    imageio.mimsave(filename, frames, "GIF", duration=duration)

def generate_comparison_video(model, val_dataset, video_dir, video_filename, frames_per_example=10, num_examples=5):
    # Select a few random examples from the dataset.
    examples = val_dataset[np.random.choice(range(len(val_dataset)), size=num_examples)]
    print("examples", examples.shape)

    # Initialize the video frames list.
    video_frames = []

    # Initialize the predicted_videos list.
    predicted_videos = []

    # Create a folder for storing GIFs
    gif_folder = os.path.join(video_dir, "gifs")
    os.makedirs(gif_folder, exist_ok=True)

    # Iterate over the examples and predict the frames.
    for ex_idx, example in enumerate(examples):
        # Pick the first/last frames_per_example frames from the example.
        frames = example[:frames_per_example, ...]
        original_frames = example[frames_per_example:, ...]
        new_predictions = np.zeros(shape=(frames_per_example, *frames[0].shape))

        print("frames", frames.shape)
        print("original_frames", original_frames.shape)

        # Predict a new set of frames_per_example frames.
        for i in range(frames_per_example):
            # Extract the model's prediction and post-process it.
            frames = example[:frames_per_example + i + 1, ...]
            new_prediction = model.predict(np.expand_dims(frames, axis=0))
            new_prediction = np.squeeze(new_prediction, axis=0)
            predicted_frame = np.expand_dims(new_prediction[-1, ...], axis=0)

            # Extend the set of prediction frames.
            new_predictions[i] = predicted_frame

        print("new_predictions", new_predictions.shape)

        # Create and save GIFs for each of the ground truth/prediction images.
        for frame_set, prefix in zip([original_frames, new_predictions], ["original", "prediction"]):
            # Convert the frames to uint8 and scale to 0-255 range.
            current_frames = np.squeeze(frame_set)
            current_frames = current_frames * 255  # No es necesario convertir a uint8 en este punto
            current_frames = current_frames.astype(np.uint8)
            print("current_frames", current_frames.shape)

            # Construct a GIF from the frames.
            gif_filename = os.path.join(gif_folder, f"{prefix}_frames_{ex_idx}.gif")  
            imageio.mimsave(gif_filename, current_frames, "GIF", duration=200) 
            predicted_videos.append(gif_filename)
            
            # Save each frame of the GIF as an image
            for i, frame in enumerate(current_frames):
                image_filename = os.path.join(gif_folder, f"{prefix}_frame_{ex_idx}_{i}.png")
                plt.imsave(image_filename, frame)

    # Create an empty list to store the combined frames
    combined_frames = []

    # Load the frames of the original and prediction GIFs for all examples
    original_frames = []
    prediction_frames = []

    # Iterate over the examples and load the frames
    for ex_idx in range(num_examples):
        print("ex_idx", ex_idx)
        # Load the frames of the original GIF
        original_gif_path = os.path.join(gif_folder, f"original_frames_{ex_idx}.gif")
        print("Loading original GIF:", original_gif_path)
        with imageio.get_reader(original_gif_path) as original_reader:
            for frame in original_reader:
                original_frames.append(frame)

        # Load the frames of the prediction GIF
        prediction_gif_path = os.path.join(gif_folder, f"prediction_frames_{ex_idx}.gif")
        print("Loading prediction GIF:", prediction_gif_path)
        with imageio.get_reader(prediction_gif_path) as prediction_reader:
            for frame in prediction_reader:
                prediction_frames.append(frame)

    # Combine the frames in the first row (original GIFs) with a separation row
    combined_frames_row1 = np.concatenate(original_frames, axis=1)
    # separation_row = np.zeros((combined_frames_row1.shape[0], 10, 3), dtype=np.uint8)
    # combined_frames_row1_with_separation = np.concatenate([combined_frames_row1, separation_row], axis=1)

    # Combine the frames in the second row (prediction GIFs)
    combined_frames_row2 = np.concatenate(prediction_frames, axis=1)

    # Combine the rows vertically
    combined_frames_example = np.concatenate([combined_frames_row1, combined_frames_row2], axis=0)

    # Append the combined frames of the current example to the list
    combined_frames.append(combined_frames_example)

    print("combined_frames_row1", combined_frames_row1.shape)

    # Convert the list of combined frames to a numpy array
    combined_frames = np.array(combined_frames)
    print("combined_frames", combined_frames.shape)

    # Reshape the combined frames to match the expected format for GIF writing
    combined_frames = combined_frames.transpose((0, 2, 1, 3)).reshape((-1, combined_frames.shape[2], combined_frames.shape[1], combined_frames.shape[3]))
    print("combined_frames reshape", combined_frames.shape)

    # Create a GIF file path
    gif_filename = os.path.join(video_dir, video_filename)

    # Save the combined frames as a GIF
    imageio.mimsave(gif_filename, combined_frames, "GIF", duration=0.2)  # Adjust the duration as needed



def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--gpu-id', type=int, default=0, help='the GPU device ID')
    p.add_argument('--m', type=int, default=0, help="The model architecture. 0: ConvLSTM, 1: Bi-directional ConvLSTM and U-Net")
    p.add_argument('--ts', type=float, default=.8, help='the train split percentage')
    p.add_argument('--bz', type=int, default=4, help='the GPU batch size')
    p.add_argument('--epochs', type=int, default=50, help='the deep learning epochs')
    p.add_argument('--w', type=int, default=64, help='the width')
    p.add_argument('--h', type=int, default=64, help='the height')
    p.add_argument('--type', type=str, default="monolayer", help='the cell type')
    p.add_argument('--prefix', type=str, default="synthetic")


    args = p.parse_args()
    results_dir = f"demo/next-image/{args.prefix}/"
    os.makedirs(results_dir, exist_ok=True)

    W.utils.set_gpu(args.gpu_id)
   
    dataset = W.dataset.load_images(base_dir="demo/", image_type=args.type, remove_first_frame=args.type=="monolayer" or args.type=="real_monolayer", resize_width=args.w, resize_height=args.h)

    # dataset = W.dataset.load_images(base_dir="demo/", image_type="monolayer", remove_first_frame=True, resize_width=64, resize_height=64)

    print(dataset.shape)

    # Split the dataset
    train_dataset, val_dataset = W.dataset.split_dataset(dataset, args.ts)

    # Normalize the data
    train_dataset = W.dataset.normalize_data(train_dataset)
    val_dataset = W.dataset.normalize_data(val_dataset)

    # Create shifted frames
    x_train, y_train = W.dataset.create_shifted_frames(train_dataset)
    x_val, y_val = W.dataset.create_shifted_frames(val_dataset)

    # Inspect the dataset
    print("Training Dataset Shapes: " + str(x_train.shape) + ", " + str(y_train.shape))
    print("Validation Dataset Shapes: " + str(x_val.shape) + ", " + str(y_val.shape))

    # Visualize and save an example
    example_index = np.random.choice(range(len(train_dataset)))
    W.dataset.visualize_example(train_dataset, example_index, f'{results_dir}example_{example_index}.png')
    W.dataset.frames_to_video(train_dataset, example_index, f'{results_dir}example_{example_index}.mp4')

    print("input shape:", x_train.shape)
    # model = W.model.create_model(x_train.shape[2:], architecture=0, num_layers=10)

    model = W.model.create_model(x_train.shape[2:], architecture=args.m, num_layers=5)
    
    # Define the checkpoint path and filename
    checkpoint_path = f"{results_dir}best_model.h5"

    # Define the ModelCheckpoint callback
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    )

    # Define some callbacks to improve training.
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=25)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)

    # Define modifiable training hyperparameters.
    epochs = args.epochs
    batch_size = args.bz


    #utils.limit_gpu_memory(args.gpu_id, 1024*8)
    # Fit the model to the training data.
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping, reduce_lr, checkpoint],
    )

    best_model = load_model(f"{results_dir}best_model.h5")

    total_frames_to_predict = val_dataset.shape[1]//2
    print("total images",total_frames_to_predict)

    for i in range(3):
        predict_and_visualize(best_model, val_dataset, results_dir, total_frames_to_predict, example_index=i)
    # create_gifs(best_model, x_val, results_dir, last_frames_number=x_val.shape[1]//2)
    
    # final_results_video = f"{results_dir}video/"
    # os.makedirs(final_results_video, exist_ok=True)
    # # generate_comparison_video(best_model, val_dataset, final_results_video, "video_result.gif", frames_per_example=val_dataset.shape[1]//2, num_examples=5)
    # generate_comparison_video(best_model, val_dataset, final_results_video, "video_result.gif", frames_per_example=total_frames_to_predict, num_examples=3)



if __name__ == "__main__":
    main()

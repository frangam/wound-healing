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
from tqdm import tqdm


import matplotlib.pyplot as plt
from tensorflow import keras

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model


import woundhealing as W

def predict_and_visualize(model, val_dataset, save_path, num_frames_initial=2, num_frames_to_show=4, example_index=0, thr_whit=0):
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
    frames = example[:num_frames_initial, ...]
    original_frames = example[num_frames_initial:, ...]
    h, w = frames.shape[1], frames.shape[2]
    new_predictions = []

    for i in tqdm(range(num_frames_to_show), "Predicting..."):
        print("frames shape", np.array(frames).shape)
        # Check if there are enough frames in the example
        if num_frames_initial + i + 1 <= len(example):
            frames = example[: num_frames_initial + i + 1, ...]
        else:
            # If not, create a black image and add it to frames
            black_image = np.zeros((1, h, w, 3))  # Added extra dimension
            frames = np.concatenate((frames, black_image), axis=0)
        
        new_prediction = model.predict(np.expand_dims(frames, axis=0))
        new_prediction = np.squeeze(new_prediction, axis=0)
        predicted_frame = np.expand_dims(new_prediction[-1, ...], axis=0)
        predicted_frame = np.squeeze(predicted_frame)
        print("predicted_frame", predicted_frame.shape)
        print(predicted_frame.dtype)
        if len(predicted_frame.shape) == 3 and predicted_frame.shape[-1] != 1:
            predicted_frame = cv2.cvtColor(predicted_frame, cv2.COLOR_BGR2GRAY)
        # else:
        #     mask_gray = cv2.normalize(src=predicted_frame, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        #     img_uint8 = mask_gray.astype(np.uint8)
        #     imgRGB = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
        #     predicted_frame = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2GRAY)
        # predicted_frame = predicted_frame.astype(np.uint8)

        if thr_whit>=0:
            predicted_frame[predicted_frame > thr_whit] = 255
        predicted_frame /= 255.0
        new_predictions.append(predicted_frame)

        orig_path = f"{save_path}original/"
        os.makedirs(orig_path, exist_ok=True)
        image_filename = os.path.join(orig_path, f"original_frame_{example_index}_{i}.png")

        if i < len(original_frames):
            plt.imsave(image_filename, original_frames[i], cmap="gray")
        else:
            black_image = Image.new('L', (w, h))
            black_image.save(image_filename)
        
        pred_path = f"{save_path}prediction/"
        os.makedirs(pred_path, exist_ok=True)
        image_filename = os.path.join(pred_path, f"prediction_frame_{example_index}_{i}.png")
        plt.imsave(image_filename, predicted_frame, cmap="gray")

    fig, axes = plt.subplots(2, num_frames_to_show, figsize=(num_frames_to_show*2, 4))

    for idx, ax in enumerate(axes[0]):
        if idx < len(original_frames):
            ax.imshow(np.squeeze(original_frames[idx]), cmap="gray")
        else:
            ax.imshow(np.zeros((h, w)), cmap="gray")
        ax.set_title(f"Original Frame {num_frames_initial + idx  + 1}")
        ax.axis("off")

    for idx, ax in enumerate(axes[1]):
        ax.imshow(np.squeeze(new_predictions[idx]), cmap="gray")
        ax.set_title(f"Predicted Frame {num_frames_initial+ idx  + 1}")
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
    '''
    Example of runs:
    -synthetic images:
    ./nextimage.py --epochs 10 --type synth_monolayer --prefix synthetic
    ./nextimage.py --epochs 100 --layers 5 --bz 4 --m 0

    ./nextimage.py --just-predict

    - real images:
    ./nextimage.py --epochs 100 --fine-tune --type real_monolayer --prefix real

    '''
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--gpu-id', type=int, default=0, help='the GPU device ID')
    p.add_argument('--type', type=str, default="synth_monolayer", help='the cell type: synth_monolayer, real_monolayer')
    p.add_argument('--prefix', type=str, default="synthetic")
    p.add_argument('--just-predict', action='store_true', help="Flag to indicate just predict and not training")
    p.add_argument('--fine-tune', action='store_true', help="Flag to indicate fine-tuning")
    p.add_argument('--fine-tune-model', type=str, default="results/next-image/synthetic/best_model.h5", help='path to the best model for fine-tuning')
    p.add_argument('--m', type=int, default=0, help="The model architecture. 0: ConvLSTM, 1: Bi-directional ConvLSTM and U-Net")
    p.add_argument('--layers', type=int, default=10, help="The number of hidden layers")
    p.add_argument('--ts', type=float, default=.8, help='the train split percentage')
    p.add_argument('--bz', type=int, default=32, help='the GPU batch size')
    p.add_argument('--epochs', type=int, default=50, help='the deep learning epochs')
    p.add_argument('--patience', type=int, default=30, help='the patience hyperparameter')
    p.add_argument('--w', type=int, default=64, help='the width')
    p.add_argument('--h', type=int, default=64, help='the height')
    p.add_argument('--th-white', type=int, default=.85, help='the threshold to draw with wite color (greater than this value)')
    p.add_argument('--start-frame', type=int, default=0, help='the start frame to predict')
    p.add_argument('--frames-pred', type=int, default=7, help='the number of frames to predict')




    args = p.parse_args()
    results_dir = f"results/next-image/{args.prefix}/"
    os.makedirs(results_dir, exist_ok=True)

    W.utils.set_gpu(args.gpu_id)
   
    # dataset = W.dataset.load_images(base_dir="data/", image_type=args.type, remove_first_frame=args.type=="synth_monolayer" or args.type=="real_monolayer", resize_width=args.w, resize_height=args.h)
    dataset = W.dataset.load_images(base_dir="data/", image_type=args.type, remove_first_frame=False, resize_width=args.w, resize_height=args.h)

    # dataset = W.dataset.load_images(base_dir="data/", image_type="monolayer", remove_first_frame=True, resize_width=64, resize_height=64)

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

    if args.fine_tune and args.fine_tune_model != "":
        model = load_model(args.fine_tune_model)
        print("loading pre-trained model:", args.fine_tune_model)
    else:
        model = W.model.create_model(x_train.shape[2:], architecture=args.m, num_layers=args.layers)

    if not args.just_predict:
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
        early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5)

        # Define modifiable training hyperparameters.
        epochs = args.epochs
        batch_size = args.bz

        #utils.limit_gpu_memory(args.gpu_id, 1024*8)
        # Fit the model to the training data.
        history = model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=[early_stopping, reduce_lr, checkpoint],
        )

        history_dict = history.history
        loss_values = history_dict["loss"]
        val_loss_values = history_dict["val_loss"]
        epochs = range(1, len(loss_values) + 1)

        plt.plot(epochs, loss_values, color="blue", marker="o", linestyle="-", label="Training loss")
        plt.plot(epochs, val_loss_values, color="red", linestyle="-", label="Validation loss")

        plt.title("Training and validation loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"results/{args.prefix}_loss_plot.png")  # Guardar el gráfico en un archivo PNG
        plt.show()

    
    best_model = load_model(f"{results_dir}best_model.h5")

    # total_frames_to_predict = (val_dataset.shape[1]//2 )
    print("total images",args.frames_pred)

    for i in range(len(x_val)):
        predict_and_visualize(best_model, val_dataset, results_dir,num_frames_initial=args.start_frame, num_frames_to_show=args.frames_pred, example_index=i, thr_whit=args.th_white)
    # create_gifs(best_model, x_val, results_dir, last_frames_number=x_val.shape[1]//2)
    
    # final_results_video = f"{results_dir}video/"
    # os.makedirs(final_results_video, exist_ok=True)
    # # generate_comparison_video(best_model, val_dataset, final_results_video, "video_result.gif", frames_per_example=val_dataset.shape[1]//2, num_examples=5)
    # generate_comparison_video(best_model, val_dataset, final_results_video, "video_result.gif", frames_per_example=total_frames_to_predict, num_examples=3)



if __name__ == "__main__":
    main()

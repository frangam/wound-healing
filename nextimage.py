#!venv/bin/python3

"""
Contains functions for building a model to predict the next whound image.

(c) All rights reserved.
original authors: Francisco M. Garcia-Moreno. 2023.

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

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model


import woundhealing as W
from woundhealing.model import ssim, mse, psnr, TransformerBlock, MultiHeadSelfAttention

# def predict_and_visualize(model, val_dataset, save_path, num_frames_initial=2, num_frames_to_show=4, example_index=0, thr_whit=0, include_original=True):
#     """
#     Generate and visualize predicted frames.

#     Parameters:
#     - model (keras.Model): The trained model.
#     - val_dataset (np.array): The validation dataset.
#     - save_path (str): The path to save the visualization.
#     - num_frames_to_show (int): The number of frames to show in the visualization. 
#       Defaults to 6.
#     - example_index (int): The index of the example from the validation dataset to visualize. 
#       Defaults to 0.
#     """
#     example = val_dataset[example_index, ...] 
#     frames = example[:num_frames_initial, ...]
#     original_frames = example[num_frames_initial:, ...]
#     h, w = frames.shape[1], frames.shape[2]
#     new_predictions = []

#     for i in tqdm(range(num_frames_to_show), "Predicting..."):
       
#         print("frames to predict", frames.shape)
#         new_prediction = model.predict(np.expand_dims(frames, axis=0))
#         print("new_prediction", new_prediction.shape)
#         new_prediction = np.squeeze(new_prediction, axis=0)
#         print("new_prediction squeeze", new_prediction.shape)
#         predicted_frame = np.expand_dims(new_prediction[-1, ...], axis=0) #take last frame predicted
#         print("predicted_frame", predicted_frame.shape)
#         sq_predicted_frame = np.squeeze(predicted_frame)
#         # print(sq_predicted_frame)
#         print("squeeze predicted_frame", sq_predicted_frame.shape)
#         # print("squeeze, tiene pixeles que no son negros?", np.any(sq_predicted_frame != 0))

#         if not include_original:
#             frames = np.concatenate((frames, predicted_frame), axis=0)
#         else:
#             frames = example[: num_frames_initial + i + 1, ...]

#         new_predictions.append(sq_predicted_frame)
        
       
    
#     inc = "incremental" if include_original else "sequential"
#     orig_path = f"{save_path}{inc}/original/"
#     pred_path = f"{save_path}{inc}/prediction/"
#     os.makedirs(orig_path, exist_ok=True)
#     os.makedirs(pred_path, exist_ok=True)
#     fig, axes = plt.subplots(2, num_frames_to_show, figsize=(num_frames_to_show*2, 4))
#     for i in range(len(new_predictions)):
#         # Guardar el frame predicho
#         predicted_frame = new_predictions[i]
#         predicted_frame = predicted_frame / np.max(predicted_frame) *255  # Ajustar los valores al rango de 0 a 255
#         predicted_frame = predicted_frame.astype(np.uint8)  # Convertir los valores a tipo de datos de 8 bits sin signo
#         predicted_frame = Image.fromarray(predicted_frame, mode="L")
#         image_filename = os.path.join(pred_path, f"prediction_frame_{example_index}_{i}.png")        
#         predicted_frame.save(image_filename)

#         # Guardar el frame original si está disponible
#         if i < len(original_frames):
#             original_frame = original_frames[i]
#         else:
#             original_frame = np.zeros_like(predicted_frame)  # Crear una imagen negra del mismo tamaño
#         original_frame = original_frame / np.max(original_frame) * 255         # Ajustar los valores al rango de 0 a 255
#         original_frame = np.squeeze(original_frame)     # Eliminar la dimensión adicional si existe
#         original_frame = original_frame.astype(np.uint8)         # Convertir los valores a tipo de datos de 8 bits sin signo
#         image = Image.fromarray(original_frame, mode="L") # Crear una imagen PIL
#         image_filename = os.path.join(orig_path, f"original_frame_{example_index}_{i}.png")
#         image.save(image_filename)

#         # Visualizar los frames en la figura
#         ax_original = axes[0, i]
#         ax_predicted = axes[1, i]

#         # Visualizar el frame original
#         ax_original.imshow(original_frame, cmap="gray")
#         ax_original.set_title(f"Original Frame {num_frames_initial + i + 1}")
#         ax_original.axis("off")

#         # Visualizar el frame predicho
#         ax_predicted.imshow(predicted_frame, cmap="gray")
#         ax_predicted.set_title(f"Predicted Frame {num_frames_initial + i + 1}")
#         ax_predicted.axis("off")

#     # Guardar la figura con todas las predicciones
#     p = f"{save_path}{inc}/all_predictions_in_a_single_image/"
#     os.makedirs(p, exist_ok=True)
#     plt.savefig(f"{p}predictions_{example_index}")
#     plt.close()
    
# def predict_and_visualize(model, x_val, y_val, data_type, save_path, example_index=0, include_original=True):
#     """
#     Generate and visualize predicted frames.

#     Parameters:
#     - model (keras.Model): The trained model.
#     - x_val (np.array): The validation dataset with shifted frames.
#     - y_val (np.array): The corresponding next frames for each shifted frame in x_val.
#     - save_path (str): The path to save the visualization.
#     - example_index (int): The index of the example from the validation dataset to visualize. 
#       Defaults to 0.
#     """
#     example_x = x_val[example_index, ...]
#     example_y = y_val[example_index, ...]
#     frames = example_x.copy()

#     # Realizar la predicción
#     new_prediction = model.predict(np.expand_dims(frames, axis=0))
#     new_prediction = np.squeeze(new_prediction, axis=0)
#     predicted_frame = np.expand_dims(new_prediction[-1, ...], axis=0)  # Take last frame predicted
#     sq_predicted_frame = np.squeeze(predicted_frame)
    
#     # Guardar las imágenes
#     inc = "incremental" if include_original else "sequential"
#     orig_path = f"{save_path}{inc}/{data_type}/original/"
#     pred_path = f"{save_path}{inc}/{data_type}/prediction/"
#     os.makedirs(orig_path, exist_ok=True)
#     os.makedirs(pred_path, exist_ok=True)

#     # Guardar el frame siguiente (example_y) con el nombre basado en el timestep
#     timestep = example_x.shape[0] + 1
#     original_frame = example_y[0]
#     original_frame = (original_frame / np.max(original_frame) * 255).astype(np.uint8)
#     original_frame = np.squeeze(original_frame)  # Ensure 2D
#     if original_frame.ndim == 3:
#         original_frame = original_frame[:, :, 0]
#     original_image = Image.fromarray(original_frame, mode="L")
#     original_image.save(os.path.join(orig_path, f"original_frame_{timestep}_{example_index}.png"))

#     # Guardar el frame predicho con el nombre basado en el timestep
#     predicted_frame = (sq_predicted_frame / np.max(sq_predicted_frame) * 255).astype(np.uint8)
#     if predicted_frame.ndim == 3:
#         predicted_frame = predicted_frame[:, :, 0]
#     predicted_image = Image.fromarray(predicted_frame, mode="L")
#     predicted_image.save(os.path.join(pred_path, f"prediction_frame_{timestep}_{example_index}.png"))
    
#     # Crear la figura para visualizar
#     num_frames_to_show = example_x.shape[0] + 1
#     fig, axes = plt.subplots(2, num_frames_to_show, figsize=(num_frames_to_show * 2, 4))
    
#     for i in range(example_x.shape[0]):
#         frame = example_x[i]
#         frame = (frame / np.max(frame) * 255).astype(np.uint8)
#         frame = np.squeeze(frame)  # Ensure 2D
#         if frame.ndim == 3:
#             frame = frame[:, :, 0]
#         axes[0, i].imshow(frame, cmap="gray")
#         axes[0, i].set_title(f"Input Frame {i + 1}")
#         axes[0, i].axis("off")

#     # Mostrar el frame original (next frame)
#     axes[0, num_frames_to_show - 1].imshow(original_frame, cmap="gray")
#     axes[0, num_frames_to_show - 1].set_title("Next Frame")
#     axes[0, num_frames_to_show - 1].axis("off")

#     # Dejar un espacio en la fila de predicciones
#     for i in range(num_frames_to_show - 1):
#         axes[1, i].axis("off")

#     # Mostrar el frame predicho justo debajo del frame original
#     axes[1, num_frames_to_show - 1].imshow(predicted_frame, cmap="gray")
#     axes[1, num_frames_to_show - 1].set_title("Predicted Frame")
#     axes[1, num_frames_to_show - 1].axis("off")
    
#     # Guardar la figura con todas las predicciones
#     os.makedirs(f"{save_path}{inc}/{data_type}/all_predictions_in_a_single_image/", exist_ok=True)
#     plt.tight_layout()
#     plt.savefig(f"{save_path}{inc}/{data_type}/all_predictions_in_a_single_image/predictions_{example_index}.png", bbox_inches='tight', pad_inches=0)
#     plt.close()

def predict_and_visualize(model, x_val, y_val, ids_val, data_type, save_path, example_index=0, include_original=True):
    """
    Generate and visualize predicted frames.

    Parameters:
    - model (keras.Model): The trained model.
    - x_val (np.array): The validation dataset with shifted frames.
    - y_val (np.array): The corresponding next frames for each shifted frame in x_val.
    - ids_val (np.array): The sequence IDs corresponding to each sequence in x_val.
    - save_path (str): The path to save the visualization.
    - example_index (int): The index of the example from the validation dataset to visualize.
    - include_original (bool): Whether to include the original frames for prediction.
    """
    example_x = x_val[example_index, ...]
    frames = example_x.copy()

    # Determinar el número de predicciones necesarias
    sequence_id = ids_val[example_index]
    sequence_indices = np.where(ids_val == sequence_id)[0]
    start_index = np.where(sequence_indices == example_index)[0][0]
    num_predictions_needed = 1 if include_original else len(sequence_indices) - start_index

    predictions = []

    # Crear directorios para guardar las imágenes
    inc = "incremental" if include_original else "sequential"
    orig_path = f"{save_path}{inc}/{data_type}/original/"
    pred_path = f"{save_path}{inc}/{data_type}/prediction/"
    os.makedirs(orig_path, exist_ok=True)
    os.makedirs(pred_path, exist_ok=True)

    # Crear la figura para visualizar
    num_frames_to_show = example_x.shape[0] + (1 if include_original else num_predictions_needed)
    fig, axes = plt.subplots(2, num_frames_to_show, figsize=(num_frames_to_show * 2, 4))

    # Visualizar frames originales
    for i in range(example_x.shape[0]):
        frame = example_x[i]
        frame = (frame / np.max(frame) * 255).astype(np.uint8)
        frame = np.squeeze(frame)  # Ensure 2D
        if frame.ndim == 3:
            frame = frame[:, :, 0]
        axes[0, i].imshow(frame, cmap="gray")
        axes[0, i].set_title(f"Input Frame {i + 1}")
        axes[0, i].axis("off")

    # Realizar predicciones y guardar frames originales y predichos
    for i in range(num_predictions_needed):
        # Realizar predicción
        new_prediction = model.predict(np.expand_dims(frames, axis=0))
        new_prediction = np.squeeze(new_prediction, axis=0)
        predicted_frame = np.expand_dims(new_prediction[-1, ...], axis=0)  # Take last frame predicted
        frames = np.concatenate((frames, predicted_frame), axis=0)[1:]  # Add predicted frame and remove first frame
        predictions.append(np.squeeze(predicted_frame))

        # Guardar el frame original
        timestep = example_x.shape[0] + i + 1
        original_frame = y_val[sequence_indices[start_index + i]]
        original_frame = (original_frame / np.max(original_frame) * 255).astype(np.uint8)
        original_frame = np.squeeze(original_frame)  # Ensure 2D
        if original_frame.ndim == 3:
            original_frame = original_frame[:, :, 0]
        original_image = Image.fromarray(original_frame, mode="L")
        original_image.save(os.path.join(orig_path, f"original_frame_{timestep}_{example_index}_{i}.png"))

        # Guardar el frame predicho
        predicted_frame = (predicted_frame / np.max(predicted_frame) * 255).astype(np.uint8)
        if predicted_frame.ndim == 3:
            predicted_frame = predicted_frame[:, :, 0]
        predicted_image = Image.fromarray(predicted_frame, mode="L")
        predicted_image.save(os.path.join(pred_path, f"prediction_frame_{timestep}_{example_index}_{i}.png"))

        # Visualizar frame original y predicho
        axes[0, example_x.shape[0] + i].imshow(original_frame, cmap="gray")
        axes[0, example_x.shape[0] + i].set_title(f"Next Frame {timestep}")
        axes[0, example_x.shape[0] + i].axis("off")

        axes[1, example_x.shape[0] + i].imshow(predicted_frame, cmap="gray")
        axes[1, example_x.shape[0] + i].set_title(f"Predicted Frame {timestep}")
        axes[1, example_x.shape[0] + i].axis("off")

    # Guardar la figura con todas las predicciones
    os.makedirs(f"{save_path}{inc}/{data_type}/all_predictions_in_a_single_image/", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{save_path}{inc}/{data_type}/all_predictions_in_a_single_image/predictions_{example_index}.png", bbox_inches='tight', pad_inches=0)
    plt.close()



    
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
    ./nextimage.py --bz 4 --patience 30 --epochs 300 --start-frame 3 --frames-pred 4 --m 9 --gpu-id 3


    ./nextimage.py --just-predict

    - real images:
    ./nextimage.py --bz 4 --patience 30 --epochs 300 --fine-tune --type real_monolayer --prefix real --start-frame 3 --frames-pred 4 --m 9 --gpu-id 3



    '''
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-g', '--gpu-id', type=int, default=0, help='the GPU device ID')
    p.add_argument('-t', '--type', type=str, default="synth_monolayer", help='the cell type: synth_monolayer, real_monolayer')
    p.add_argument('-p', '--prefix', type=str, default="synthetic")
    # p.add_argument('--inc-original', action='store_true', help="Flag to indicate include the original frames to predict the next frames")
    p.add_argument('-jp', '--just-predict', action='store_true', help="Flag to indicate just predict and not training")
    p.add_argument('-f', '--fine-tune', action='store_true', help="Flag to indicate fine-tuning")
    p.add_argument('-ft', '--fine-tune-type', action='store_true', help="Flag to indicate to use the fine-tuning specific on cell type")
    p.add_argument('--fine-tune-model', type=str, default="best_model.h5", help='path to the best model for fine-tuning')
    p.add_argument('-m', '--m', type=int, default=0, help="The model architecture. 0: ConvLSTM, 1: Bi-directional ConvLSTM and U-Net")
    p.add_argument('--layers', type=int, default=5, help="The number of hidden layers")
    p.add_argument('--ts', type=float, default=.2, help='the test split percentage')
    p.add_argument('-bz', '--bz', type=int, default=32, help='the GPU batch size')
    p.add_argument('-e', '--epochs', type=int, default=50, help='the deep learning epochs')
    p.add_argument('-pa', '--patience', type=int, default=10, help='the patience hyperparameter')
    p.add_argument('--w', type=int, default=64, help='the width')
    p.add_argument('--h', type=int, default=64, help='the height')
    p.add_argument('--th-white', type=int, default=.85, help='the threshold to draw with wite color (greater than this value)')
    p.add_argument('-sf', '--start-frame', type=int, default=2, help='the start frame to predict')
    p.add_argument('-fp', '--frames-pred', type=int, default=5, help='the number of frames to predict')
    p.add_argument('-s', '--save-real-loaded', action='store_true', help="Flag to indicate saving real data when loaded")
    p.add_argument('-c','--concat', action='store_true', help="Flag to indicate concatenating all cell types")





    args = p.parse_args()
    results_dir = f"results/next-image/{args.prefix}/model_{args.m}/"
    os.makedirs(results_dir, exist_ok=True)

    W.utils.set_gpu(args.gpu_id)

  
    # dataset = W.dataset.load_images(base_dir="data/", image_type=args.type, remove_first_frame=args.type=="synth_monolayer" or args.type=="real_monolayer", resize_width=args.w, resize_height=args.h)
    if "synthetic" in args.prefix:
        dataset = W.dataset.load_images(base_dir="data/", image_type="synth_monolayer_old", remove_first_frame=True, resize_width=args.w, resize_height=args.h, fill=False)
        dataset2 = W.dataset.load_images(base_dir="data/", image_type="synth_spheres_old", remove_first_frame=True, resize_width=args.w, resize_height=args.h, fill=True)
        # dataset3 = W.dataset.load_images(base_dir="data/", image_type="aug_synth_monolayer", remove_first_frame=False, resize_width=args.w, resize_height=args.h)
        # dataset4 = W.dataset.load_images(base_dir="data/", image_type="aug_synth_spheres", remove_first_frame=False, resize_width=args.w, resize_height=args.h)
        # dataset = np.concatenate((dataset, dataset3), axis=0)
        # dataset2 = np.concatenate((dataset2, dataset4), axis=0)
    else:
        dataset = W.dataset.load_images(base_dir="data/", image_type="real_monolayer", remove_first_frame=False, resize_width=args.w, resize_height=args.h, fill=False)
        dataset2 = W.dataset.load_images(base_dir="data/", image_type="real_spheres", remove_first_frame=False, resize_width=args.w, resize_height=args.h, fill=True)

    if args.save_real_loaded:
        W.utils.save_image_dataset_loaded(dataset, "real_monolayer")
        W.utils.save_image_dataset_loaded(dataset2, "real_spheres")

        # dataset3 = W.dataset.load_images(base_dir="data/", image_type="aug_real_monolayer", remove_first_frame=False, resize_width=args.w, resize_height=args.h, fill=False)
        # dataset4 = W.dataset.load_images(base_dir="data/", image_type="aug_real_spheres", remove_first_frame=False, resize_width=args.w, resize_height=args.h, fill=False)
        # dataset = np.concatenate((dataset, dataset3), axis=0)
        # dataset2 = np.concatenate((dataset2, dataset4), axis=0)
    print("dataset (monolayer) shape:", dataset.shape)
    print("dataset (spheres) shape:", dataset2.shape)
    if args.concat:
        labels1= np.repeat(0, dataset.shape[0])
        labels2= np.repeat(1, dataset2.shape[0])
        dataset = np.concatenate((dataset, dataset2), axis=0)
        labels = np.concatenate((labels1, labels2), axis=0)

        print('Primeras etiquetas:')
        print(labels[:12])

        print('Últimas etiquetas:')
        print(labels[-12:])
    else:
        labels = None
        if "spheres" in args.type:
            dataset = dataset2

    print("complete dataset:", dataset.shape)

    dataset = W.dataset.normalize_data(dataset)# Normalize the data
    dataset = dataset[..., :1] #to gray
    x, y, seq_ids,labels = W.dataset.create_shifted_frames(dataset,labels) # Split the dataset
    print("Shifted dataset X:", x.shape, " y:", y.shape)
    x_train, x_val, x_test, y_train, y_val, y_test, train_labels, val_labels, test_labels, train_ids, val_ids, test_ids = W.dataset.split_dataset(x, y, labels, seq_ids, val_ratio=args.ts, test_ratio=args.ts)

    # Imprimir los IDs únicos
    unique_train_ids = np.unique(train_ids)
    unique_val_ids = np.unique(val_ids)
    unique_test_ids = np.unique(test_ids)

    print("IDs únicos en el conjunto de entrenamiento:", unique_train_ids)
    print("IDs únicos en el conjunto de validación:", unique_val_ids)
    print("IDs únicos en el conjunto de prueba:", unique_test_ids)

    # Verificar solapamiento
    print("Solapamiento entre entrenamiento y validación:", np.intersect1d(unique_train_ids, unique_val_ids))
    print("Solapamiento entre entrenamiento y prueba:", np.intersect1d(unique_train_ids, unique_test_ids))
    print("Solapamiento entre validación y prueba:", np.intersect1d(unique_val_ids, unique_test_ids))

    # # Split the dataset
    # train_dataset, val_dataset, train_labels, val_labels = W.dataset.split_dataset(dataset, labels, args.ts, seed=33)
    # train_dataset, test_dataset, train_labels, test_labels = W.dataset.split_dataset(train_dataset, train_labels, args.ts, seed=33)

    # # Normalize the data
    # train_dataset = W.dataset.normalize_data(train_dataset)
    # val_dataset = W.dataset.normalize_data(val_dataset)
    # test_dataset = W.dataset.normalize_data(test_dataset)

    # train_dataset = train_dataset[..., :1] #to gray
    # val_dataset = val_dataset[..., :1] #to gray
    # test_dataset = test_dataset[..., :1] #to gray

    # # Create shifted frames
    # x_train, y_train = W.dataset.create_shifted_frames(train_dataset)
    # x_val, y_val = W.dataset.create_shifted_frames(val_dataset)
    # x_test, y_test = W.dataset.create_shifted_frames(val_dataset)


    # Inspect the dataset
    print("Training Dataset Shapes: " + str(x_train.shape) + ", " + str(y_train.shape))
    print("Validation Dataset Shapes: " + str(x_val.shape) + ", " + str(y_val.shape))
    print("Test Dataset Shapes: " + str(x_test.shape) + ", " + str(y_test.shape))

    # Visualize and save an example
    example_index = np.random.choice(range(len(dataset)))
    os.makedirs(f'{results_dir}examples/', exist_ok=True)
    W.dataset.visualize_example(dataset, example_index, f'{results_dir}examples/example_{example_index}.png')
    W.dataset.frames_to_video(dataset, example_index, f'{results_dir}examples/example_{example_index}.mp4')

    print("input shape:", x_train.shape)
    # model = W.model.create_model(x_train.shape[2:], architecture=0, num_layers=10)

    if args.fine_tune and args.fine_tune_model != "":
        if args.fine_tune_type:
            p = "synth-monolayer" if "monolayer" in args.type else "synth-spheres"
        else: #just synthetic path
            p = "synthetic"

        model = load_model(f"results/next-image/{p}/model_{args.m}/{args.fine_tune_model}", custom_objects={'ssim': ssim, 'mse': mse, 'psnr': psnr, 'MultiHeadSelfAttention': MultiHeadSelfAttention, 'TransformerBlock': TransformerBlock})
        
        print("loading pre-trained model:", args.fine_tune_model)
    else:
        model = W.model.create_model(x_train.shape, architecture=args.m, num_layers=args.layers)

    if not args.just_predict:
        # Define the checkpoint path and filename
        checkpoint_path = f"{results_dir}best_model.h5"

        # Define the ModelCheckpoint callback
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor="val_ssim", #"val_loss",
            save_best_only=True,
            mode="max", #"min",
            verbose=1
        )

        # Define some callbacks to improve training.
        early_stopping = keras.callbacks.EarlyStopping(monitor="val_ssim", patience=args.patience)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_ssim", patience=5)

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

        # history_dict = history.history
        # loss_values = history_dict["loss"]
        # val_loss_values = history_dict["val_loss"]
        # epochs = range(1, len(loss_values) + 1)
        # plt.plot(epochs, loss_values, color="blue", marker="o", linestyle="-", label="Training loss")
        # plt.plot(epochs, val_loss_values, color="red", linestyle="-", label="Validation loss")
        # plt.title("Training and validation loss")
        # plt.xlabel("Epochs")
        # plt.ylabel("Loss")
        # plt.legend()
        # plt.savefig(f"results/{args.prefix}_loss_plot.png")  # Guardar el gráfico en un archivo PNG
        # plt.show()


        ##################################################################
        # plot the training loss and accuracy
        history_dict = history.history
        plt.style.use("ggplot")
        plt.figure()
        N = len(history_dict["ssim"])
        plt.plot(np.arange(0, N), history_dict["ssim"], label="ssim")
        plt.plot(np.arange(0, N), history_dict["val_ssim"], label="val_ssim")
        plt.plot(np.arange(0, N), history_dict["mse"], label="mse")
        plt.plot(np.arange(0, N), history_dict["val_mse"], label="val_mse")
        plt.title("Training and Validation SSIM and MSE")
        plt.xlabel("Epoch #")
        plt.ylabel("SSIM/MSE")
        plt.legend(loc="best", bbox_to_anchor=(0.5, 0., 0.5, 0.5))
        plt.savefig(f"{results_dir}/{args.prefix}_ssim_mse_history_plot.png", dpi=300)


        ##################################################################

    if args.m == 19 or args.m==20:
        best_model = model
    else:
        best_model = tf.keras.models.load_model(f"{results_dir}best_model.h5", custom_objects={'ssim': ssim, 'mse': mse, 'psnr': psnr, 'MultiHeadSelfAttention': MultiHeadSelfAttention, 'TransformerBlock': TransformerBlock})
    

    # total_frames_to_predict = (val_dataset.shape[1]//2 )
    print("total images",args.frames_pred)


    # Generating predictions
    # for i in range(len(x_val)):
    #     print("generating validation predictions, id:", i)
    #     predict_and_visualize(best_model, x_val, y_val, "validation", results_dir, example_index=i, include_original=True)
    #     predict_and_visualize(best_model, x_val, y_val, "validation", results_dir, example_index=i, include_original=False)

    for i in range(len(x_test)-10, len(x_test)):
        print("generating test predictions, id:", i)
        predict_and_visualize(best_model, x_test, y_test, test_ids, "test", results_dir, example_index=i, include_original=True)
        predict_and_visualize(best_model, x_test, y_test, test_ids, "test", results_dir, example_index=i, include_original=False)



    # create_gifs(best_model, x_val, results_dir, last_frames_number=x_val.shape[1]//2)
    
    # final_results_video = f"{results_dir}video/"
    # os.makedirs(final_results_video, exist_ok=True)
    # # generate_comparison_video(best_model, val_dataset, final_results_video, "video_result.gif", frames_per_example=val_dataset.shape[1]//2, num_examples=5)
    # generate_comparison_video(best_model, val_dataset, final_results_video, "video_result.gif", frames_per_example=total_frames_to_predict, num_examples=3)



if __name__ == "__main__":
    main()

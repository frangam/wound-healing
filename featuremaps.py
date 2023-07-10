#!venv/bin/python3

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
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import keras
import matplotlib as mpl

import IPython.display as display
import PIL.Image
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer

from woundhealing.model import ssim, mse, psnr, TransformerBlock, MultiHeadSelfAttention


import woundhealing as W


# Download an image and read it into a NumPy array.
def download(url, max_dim=None):
    name = url.split('/')[-1]
    image_path = tf.keras.utils.get_file(name, origin=url)
    img = PIL.Image.open(image_path)
    if max_dim:
        img.thumbnail((max_dim, max_dim))
    return np.array(img)


# Normalize an image
def deprocess(img):
    img = 255*(img + 1.0)/2.0
    return tf.cast(img, tf.uint8)

# Display an image
def show(img):
    display.display(PIL.Image.fromarray(np.array(img)[0]))


def calc_loss(img, model):
    # Pass forward the image through the model to retrieve the activations.
    # Converts the image into a batch of size 1.
    img_batch = tf.expand_dims(img, axis=0)
    layer_activations = model(img_batch)
    if len(layer_activations) == 1:
        layer_activations = [layer_activations]

    losses = []
    for act in layer_activations:
        loss = tf.math.reduce_mean(act)
        losses.append(loss)

    return  tf.reduce_sum(losses)

class DeepDream(tf.Module):
    def __init__(self, model):
        self.model = model

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None, None,None,3], dtype=tf.float32),
            tf.TensorSpec(shape=[], dtype=tf.int32),
            tf.TensorSpec(shape=[], dtype=tf.float32),)
    )
    def __call__(self, img, steps, step_size):
        print("Tracing")
        loss = tf.constant(0.0)
        img_3D = img[0, ...]
        print("img_3D", img_3D.shape)
        for n in tf.range(steps):
            img = tf.expand_dims(img_3D, axis=0)
            with tf.GradientTape() as tape:
                # This needs gradients relative to `img`
                # `GradientTape` only watches `tf.Variable`s by default
                
                tape.watch(img)
                loss = calc_loss(img, self.model)
                print("loss", loss)

            # Calculate the gradient of the loss with respect to the pixels of the input image.
            gradients = tape.gradient(loss, img)
            print("gradients", gradients, "\nshape:", gradients.shape)

            # Normalize the gradients.
            gradients /= tf.math.reduce_std(gradients) + 1e-8 
            
            # In gradient ascent, the "loss" is maximized so that the input image increasingly "excites" the layers.
            # You can update the image by directly adding the gradients (because they're the same shape!)
            img_3D = img_3D + gradients[0,...]*step_size
            print("img_3D", img_3D.shape)
            img_3D = tf.clip_by_value(img_3D, -1, 1)

        return loss, img_3D

def run_deep_dream_simple(deepdream, img, steps=100, step_size=0.01):
    # Convert from uint8 to the range expected by the model.
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    img = tf.convert_to_tensor(img)
    step_size = tf.convert_to_tensor(step_size)
    steps_remaining = steps
    step = 0
    while steps_remaining:
        if steps_remaining>100:
            run_steps = tf.constant(100)
        else:
            run_steps = tf.constant(steps_remaining)
        
        steps_remaining -= run_steps
        step += run_steps

        loss, img = deepdream(img, run_steps, tf.constant(step_size))
        
        display.clear_output(wait=True)
        show(deprocess(img))
        print ("Step {}, loss {}".format(step, loss))


    result = deprocess(img)
    display.clear_output(wait=True)

    
    show(result)

    return result

'''
Examples of run:
./featuremaps.py --type synth_monolayer --prefix synthetic --m 3
./featuremaps.py --type real_monolayer --prefix real --m 3
'''
p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument('--gpu-id', type=int, default=0, help='the GPU device ID')
p.add_argument('--prefix', type=str, default="synthetic")
p.add_argument('--type', type=str, default="synth_monolayer", help='the cell type: synth_monolayer, real_monolayer')
p.add_argument('--m', type=int, default=0, help="The model architecture. 0: ConvLSTM, 1: Bi-directional ConvLSTM and U-Net")
p.add_argument('--w', type=int, default=64, help='the width')
p.add_argument('--h', type=int, default=64, help='the height')
p.add_argument('--ts', type=float, default=.2, help='the train split percentage')
args = p.parse_args()
W.utils.set_gpu(args.gpu_id)

# dataset = W.dataset.load_images(base_dir="data/", image_type=args.type, remove_first_frame=args.type=="synth_monolayer" or args.type=="real_monolayer", resize_width=args.w, resize_height=args.h)
dataset = W.dataset.load_images(base_dir="data/", image_type=args.type, remove_first_frame=False, resize_width=args.w, resize_height=args.h)

# dataset = W.dataset.load_images(base_dir="data/", image_type="monolayer", remove_first_frame=True, resize_width=64, resize_height=64)

print(dataset.shape)

# Split the dataset
_, val_dataset = W.dataset.split_dataset(dataset, None, args.ts)
# Normalize the data
val_dataset_original = W.dataset.normalize_data(val_dataset)
val_dataset = val_dataset_original[..., :1] #to gray
# Inspect the dataset
print("Validation Dataset Shapes: " + str(val_dataset.shape))
example = val_dataset[0, ...] 
frames = example[:4, ...]


results_dir = f"results/next-image/{args.prefix}/model_{args.m}/"
model = load_model(f"{results_dir}best_model.h5", custom_objects={'ssim': ssim, 'mse': mse, 'psnr': psnr, 'MultiHeadSelfAttention': MultiHeadSelfAttention, 'TransformerBlock': TransformerBlock})
layer_names = [layer.name for layer in model.layers]
intermediate_model = keras.Model(inputs=model.input, outputs=[model.get_layer(layer_name).output for layer_name in layer_names])

print("layer_names", layer_names)
# Obtener los feature maps para la muestra de entrada
feature_maps = intermediate_model.predict(np.expand_dims(frames, axis=0))

# Visualizar y guardar los feature maps de cada capa
for layer_name, feature_map in zip(layer_names, feature_maps):
    # Solo considerar las capas convolucionales (ignorar las capas de entrada y las capas completamente conectadas)
    print("layer:", layer_name)
    # Guardar los kernels aprendidos de las capas convolucionales
    if layer_name.startswith("conv"):
        layer = model.get_layer(layer_name)
        weights = layer.get_weights()[0]
        kernel_grid_size = int(np.ceil(np.sqrt(weights.shape[3])))
        fig, ax = plt.subplots(kernel_grid_size, kernel_grid_size, figsize=(10, 10))
        for i in range(weights.shape[3]):
            if kernel_grid_size > 1:
                sub_ax = ax[i // kernel_grid_size, i % kernel_grid_size]
            else:
                sub_ax = ax
            print("weights", weights.shape)
            sub_ax.imshow(weights[:, :, 0, i], cmap="gray")  # Cambiar el índice (0) si hay más canales en la entrada
            sub_ax.axis("off")
        kernel_savepath = f'{results_dir}kernels/{layer_name}/'
        os.makedirs(kernel_savepath, exist_ok=True)
        plt.savefig(f"{kernel_savepath}kernels_{layer_name}.png")
        plt.close()

    # if layer_name.startswith('conv_lstm2d'):
        # print("here")
    # Obtener el número de filtros en la capa
    num_filters = feature_map.shape[-1]
    print("feature_map.shape", feature_map.shape, "num_filters", num_filters)
    feature_map = np.squeeze(feature_map, axis=0)
    feature_map = np.expand_dims(feature_map[-1, ...], axis=0) #take last frame predicted
    feature_map = np.squeeze(feature_map)

    # Crear una cuadrícula para visualizar los feature maps
    grid_size = int(np.ceil(np.sqrt(num_filters)))
    fig, ax = plt.subplots(grid_size, grid_size, figsize=(10, 10))

    # Visualizar cada filtro en la cuadrícula
    for i in range(num_filters):
        if grid_size > 1:
            sub_ax = ax[i // grid_size, i % grid_size]
        else:
            sub_ax = ax

        # Visualizar el filtro correspondiente
        if num_filters > 1:
            sub_ax.imshow(feature_map[:, :, i], cmap="gray")
        else:
            sub_ax.imshow(feature_map, cmap="gray")
        sub_ax.axis("off")

    savepath = f'{results_dir}feature_maps/{layer_name}/'
    os.makedirs(savepath, exist_ok=True)
    # Guardar la imagen de los feature maps de la capa
    print("Saving to...", f"{savepath}feature_maps_{layer_name}.png")
    plt.savefig(f"{savepath}feature_maps_{layer_name}.png")
    plt.close()
    # else:
    #     print("layer", layer_name, "is not a conv_lstm")


# for layer in model.layers:
#     print(layer.name)

#     # Verificar si la capa tiene filtros (por ejemplo, Conv2D)
#     if layer.name == 'conv_lstm2d' and hasattr(layer, 'filters'):
#         layer = model.get_layer(layer.name)

#         # Crear un nuevo modelo que tiene como salida la capa deseada
#         feature_map_model = tf.keras.models.Model(inputs=model.input, outputs=layer.output)

#         num_filters = layer.filters
        
#         savepath = f'{results_dir}feature_maps/{layer.name}/'
#         os.makedirs(savepath, exist_ok=True)

        
#         # Iterar sobre los filtros de la capa
#         for i in range(layer.filters):
#             # Obtener el tensor del feature map del filtro específico
#             feature_map = feature_map_model.predict(np.expand_dims(frames, axis=0))[..., i]

#             # Guardar el array NumPy como una imagen en color
#             plt.imsave(f'{savepath}feature_maps/{layer.name}_filter_{i}.png', feature_map, cmap='gray')




print("processing DeepDream")
val_dataset = val_dataset_original[..., :1] #to gray

# Maximize the activations of these layers
print([l.name for l in model.layers])
names = [f'conv_lstm2d_{i}' for i in range(1,6)]
layers = [model.get_layer(name).output for name in names]


# Create the feature extraction model
dream_model = tf.keras.Model(inputs=model.input, outputs=layers)

url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'
down_img = download(url, max_dim=500)
print("donw", np.array(down_img).shape)
# print(down_img)

original_img = val_dataset[0, 0, ...]
original_imgfloat = original_img.astype(np.float32)  # Convertir el dtype a float32
original_img_tf = tf.convert_to_tensor(original_imgfloat)  # Convertir a un tensor de TensorFlow
original_img_tf = tf.expand_dims(original_img_tf, axis=0)  # Agregar una dimensión adicional para el lote (batch)

print("validation original_img", original_img.shape)
dpdream = DeepDream(dream_model)
dream_img = run_deep_dream_simple(dpdream, img=original_img_tf, steps=100, step_size=0.01)
# dream_img_np = dream_img.numpy()
# dream_img_np = ((dream_img_np + 1) / 2) * 255
# dream_img_np = dream_img_np.astype(np.uint8)
img_pil = PIL.Image.fromarray(np.array(dream_img))

savepath = f'{results_dir}feature_maps/{names[0]}/'
os.makedirs(savepath, exist_ok=True)
img_pil.save(f'{savepath}dream_img.png')


img_pil = PIL.Image.fromarray(np.array(original_img))
img_pil.save(f'{savepath}dream_img.png')
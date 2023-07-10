#!venv/bin/python3

import os
import argparse

import imgaug.augmenters as iaa
import imgaug as ia
import numpy as np
import imageio

import woundhealing as W

'''
Examples of run

./real_augment.py -t synth_monolayer -n 4 -g 3
./real_augment.py -t synth_spheres -n 4 -g 3
./real_augment.py -t real_monolayer -n 4 -g 3
./real_augment.py -t real_spheres -n 4 -g 3

'''


p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument('-t', '--type', type=str, default="synth_monolayer", help='the cell type: synth_monolayer, real_monolayer')
p.add_argument('-n', '--number', type=int, default=10, help='the times to multiply the original images to augment: 10 x number of originals')
p.add_argument('-f', '--fill', action='store_true', help="Flag to indicate to fill empty hours with frames")
p.add_argument('-g', '--gpu-id', type=int, default=0, help='the GPU device ID')

args = p.parse_args()
W.utils.set_gpu(args.gpu_id)


WIDTH = 64
HEIGHT = 64
SEED = 33
image_type=args.type
# cada secuencia es un np.array con forma (frames, width, height, 3)
sequences = [seq for seq in W.dataset.load_images(base_dir="data/", image_type=image_type, remove_first_frame=False, resize_width=WIDTH, resize_height=HEIGHT, fill=args.fill)]

# print(sequences.shape)

# Crear el objeto augmenter
seq = iaa.Sequential([
    iaa.Fliplr(0.1), # voltea 50% de las imágenes de forma horizontal
    iaa.Affine(rotate=(-0.1, .1)) , # Rotar entre -25 y 25 grados
    iaa.PiecewiseAffine(scale=(0.001, 0.005))  # Transformación elástica: Útil para imágenes donde objetos se deforman, como imágenes biomédicas, 
], random_order=True)

# Crear un estado aleatorio y usa el mismo estado para cada imagen en una secuencia
random_state = ia.new_random_state(SEED)

sequences_augmented = [seq(images=images_seq) for _ in range(args.number) for images_seq in sequences]
sequences_augmented = np.array(sequences_augmented)
print("sequences_augmented", sequences_augmented.shape)


for i, seq in enumerate(sequences_augmented):
    for j, img in enumerate(seq):
        # Crear un nombre de archivo con base en los índices de la secuencia y la imagen
        spath = f"data/aug_{image_type}/frames_{i}/"
        os.makedirs(spath, exist_ok=True)

        filename = f"{spath}image_{j}.png"

        # Guardar la imagen
        imageio.imsave(filename, img)


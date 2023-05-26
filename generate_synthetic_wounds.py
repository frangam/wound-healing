#!venv/bin/python3
import argparse
import numpy as np
from woundhealing.synthetic import generate_wound, generate_video

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--n', type=int, default=5, help='the number of wounds to generate')
    p.add_argument('--w', type=int, default=1000, help='the image width')
    p.add_argument('--h', type=int, default=1000, help='the image height')
    args = p.parse_args()

    IMG_WIDTH = args.w
    IMG_HEIGHT = args.h
    
    MONOLAYER = [0, 3, 6, 9, 12, 24, 27]
    SPHERES = [0, 3, 6, 9, 12, 15]

    # Pick a random start point
    seed_high  = np.rint(np.random.uniform(0, IMG_HEIGHT, 1))[0].astype(np.int32)
    seed_left  = np.rint(np.random.normal(0.25 * IMG_WIDTH, 1, 1))[0].astype(np.int32)
    seed_right = np.rint(np.random.normal(0.75 * IMG_WIDTH, 1, 1))[0].astype(np.int32)

    print('Initial values:', seed_left, seed_right, seed_high)

    # Demo to generate two cell types and videos
    monolayer_wounds = generate_wound(MONOLAYER, seed_left, seed_right, seed_high, IMG_WIDTH, IMG_HEIGHT)
    generate_video(monolayer_wounds, "demo/monolayer/monolayer_video.mp4", "demo/monolayer/", IMG_WIDTH, IMG_HEIGHT)
    sphere_wounds = generate_wound(SPHERES, seed_left, seed_right, seed_high, IMG_WIDTH, IMG_HEIGHT)
    generate_video(sphere_wounds, "demo/spheres/spheres_video.mp4", "demo/spheres/", IMG_WIDTH, IMG_HEIGHT)


if __name__ == '__main__':
    main()

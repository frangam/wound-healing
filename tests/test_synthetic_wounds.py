import sys
import os
import pytest
import pandas as pd


# Agregar la ruta del directorio raíz del paquete al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from woundhealing.synthetic import generate_wound, generate_wound_dataframe, generate_video






@pytest.fixture
def seed_values():
    seed_left = 250
    seed_right = 750
    seed_high = 500
    return seed_left, seed_right, seed_high


@pytest.fixture
def image_dimensions():
    IMG_WIDTH = 1000
    IMG_HEIGHT = 1000
    return IMG_WIDTH, IMG_HEIGHT


@pytest.fixture
def cell_types():
    MONOLAYER = [0, 3, 6, 9, 12, 24, 27]
    SPHERES = [0, 3, 6, 9, 12, 15]
    return MONOLAYER, SPHERES


def test_generate_wound(seed_values, image_dimensions, cell_types):
    seed_left, seed_right, seed_high = seed_values
    IMG_WIDTH, IMG_HEIGHT = image_dimensions

    monolayer_wounds = generate_wound(cell_types[0], seed_left, seed_right, seed_high, IMG_WIDTH, IMG_HEIGHT)
    sphere_wounds = generate_wound(cell_types[1], seed_left, seed_right, seed_high, IMG_WIDTH, IMG_HEIGHT)

    assert len(monolayer_wounds) == len(cell_types[0])
    assert len(sphere_wounds) == len(cell_types[1])



def test_generate_wound_dataframe(seed_values, image_dimensions, cell_types):
    seed_left, seed_right, seed_high = seed_values
    IMG_WIDTH, IMG_HEIGHT = image_dimensions

    monolayer_df = generate_wound_dataframe(cell_types[0], seed_left, seed_right, seed_high, IMG_WIDTH, IMG_HEIGHT)
    sphere_df = generate_wound_dataframe(cell_types[1], seed_left, seed_right, seed_high, IMG_WIDTH, IMG_HEIGHT)
    
    monolayer_df['ID'] = f'Monolayer'
    monolayer_df['CellType'] = 0

    sphere_df['ID'] = f'Sphere'
    sphere_df['CellType'] = 1


    assert isinstance(monolayer_df, pd.DataFrame)
    assert isinstance(sphere_df, pd.DataFrame)

    assert monolayer_df.shape[1] == len(cell_types[0]) + 2  # +2 for ID and CellType columns
    assert sphere_df.shape[1] == len(cell_types[1]) + 2  # +2 for ID and CellType columns


def test_generate_video(seed_values, image_dimensions, cell_types):
    seed_left, seed_right, seed_high = seed_values
    IMG_WIDTH, IMG_HEIGHT = image_dimensions

    monolayer_wounds = generate_wound(cell_types[0], seed_left, seed_right, seed_high, IMG_WIDTH, IMG_HEIGHT)
    sphere_wounds = generate_wound(cell_types[1], seed_left, seed_right, seed_high, IMG_WIDTH, IMG_HEIGHT)

    # Generate videos
    monolayer_video_path = "demo/monolayer/monolayer_video.mp4"
    sphere_video_path = "demo/spheres/sphere_video.mp4"
    monolayer_image_folder = "demo/monolayer/"
    sphere_image_folder = "demo/spheres/"

    generate_video(monolayer_wounds, monolayer_video_path, monolayer_image_folder, IMG_WIDTH, IMG_HEIGHT)
    generate_video(sphere_wounds, sphere_video_path, sphere_image_folder, IMG_WIDTH, IMG_HEIGHT)

    # Validate the generated videos
    assert os.path.isfile(monolayer_video_path)
    assert os.path.isfile(sphere_video_path)
    assert os.path.isdir(monolayer_image_folder)
    assert os.path.isdir(sphere_image_folder)

    # TODO: Add additional assertions or validation steps if necessary

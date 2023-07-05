import os
import tensorflow as tf
import tifffile


MONOLAYER = [0, 3, 6, 9, 12, 24, 27]
SPHERES = [0, 3, 6, 9, 12, 18, 24]
CELL_TYPES = {0: MONOLAYER, 1: SPHERES}

def len_cell_type_time_step(celltype=0):
    try:
        ts = CELL_TYPES[celltype]
        return len(ts)
    except RuntimeError as e:
        print(f"Cell type [{celltype}] not availble [we only work with: {CELL_TYPES.keys()}]")

# Specify the GPU ID of the device you wish to use. 
# Note: The GPU ID starts at 0.
def set_gpu(gpu_id=0):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
      try:
        tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
      except RuntimeError as e:
        print(e)
    else:
        print("No GPUs available")



def limit_gpu_memory(gpu_id, memory_limit=0.8):
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        if gpu_id < len(physical_devices):
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    physical_devices[gpu_id],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)]
                )
                logical_devices = tf.config.list_logical_devices('GPU')
                print(len(physical_devices), "Physical GPUs,", len(logical_devices), "Logical GPU")
            except RuntimeError as e:
                print(e)
        else:
            print("Invalid GPU ID")
    else:
        print("No GPUs available")

def grow_gpu_memory(gpu_id=0, memory_limit=1024*8):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus[gpu_id]), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)



def get_pixel_size(tiff_path):
    with tifffile.TiffFile(tiff_path) as tif:
        tags = tif.pages[0].tags
        if 'XResolution' in tags and 'YResolution' in tags:
            x_resolution = tags['XResolution'].value
            y_resolution = tags['YResolution'].value
            if isinstance(x_resolution, tuple) and isinstance(y_resolution, tuple):
                x_resolution = x_resolution[0] / x_resolution[1]
                y_resolution = y_resolution[0] / y_resolution[1]
            if x_resolution == y_resolution:
                pixel_size = 1 / x_resolution
                if 'ResolutionUnit' in tags:
                    resolution_unit = tags['ResolutionUnit'].value
                    if resolution_unit == 2:  # Pulgadas
                        pixel_size *= 25.4  # Convertir de pulgadas a micrómetros (μm)
                    elif resolution_unit == 3:  # Centímetros
                        pixel_size *= 10  # Convertir de centímetros a micrómetros (μm)
                return pixel_size
    return None
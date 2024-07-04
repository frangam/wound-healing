#!venv/bin/python3

# import os
# import matplotlib.pyplot as plt
# import cv2
# import numpy as np
# from scipy.stats import norm

# # Define paths to the directories
# real_images_base_dir = 'data/real_segmentations/Monolayer/'
# synthetic_images_base_dir = 'data/synth_monolayer_old/'

# # Function to recursively load images from a directory and its subdirectories
# def load_images_from_directory(base_directory, real=False, num_time_stamps=7):
#     images = []
#     for root, dirs, files in os.walk(base_directory):
#         if real:
#             # Check if it's a real image directory and enter the "gray" folder
#             if "gray" in root:
#                 time_stamp_images = []  # Store images for each time-stamp
#                 for file in files:
#                     if file.endswith(".png"):
#                         img_path = os.path.join(root, file)
#                         img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#                         if img is not None:
#                             time_stamp_images.append(img)
#                 # Check if we have more than 7 images for this time-stamp
#                 if len(time_stamp_images) > num_time_stamps:
#                     time_stamp_images = time_stamp_images[:num_time_stamps]  # Keep only the first 7 images
#                 images.extend(time_stamp_images)
#         else:
#             # For synthetic images, check the number of time-stamps
#             time_stamp_images = []  # Store images for each time-stamp
#             for file in files:
#                 if file.endswith(".png"):
#                     img_path = os.path.join(root, file)
#                     img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#                     if img is not None:
#                         time_stamp_images.append(img)
#                     # Check if we have more than 7 images or if we have reached the end of the time-stamp
#                     if len(time_stamp_images) > num_time_stamps or (files.index(file) == len(files) - 1 and len(time_stamp_images) > 0):
#                         time_stamp_images = time_stamp_images[:num_time_stamps]  # Keep only the first 7 images
#                         images.extend(time_stamp_images)
#                         time_stamp_images = []  # Reset for the next time-stamp
#     return images



# # Load the real and synthetic images
# real_images = load_images_from_directory(real_images_base_dir, real=True)
# synthetic_images = load_images_from_directory(synthetic_images_base_dir)

# print(len(real_images), real_images[0].shape)
# print(len(synthetic_images), synthetic_images[0].shape)


# # Function to calculate the average pixel values and white area of images
# def calculate_metrics(images):
#     avg_values = [np.mean(image) for image in images]
#     white_areas = [calculate_white_area(image) for image in images]
#     return avg_values, white_areas

# # Function to calculate the white area in an image
# def calculate_white_area(image):
#     # _, binary_mask = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
#     # white_area = cv2.countNonZero(binary_mask)
#     _, binary_mask = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
#     white_area_per_pixel = cv2.countNonZero(binary_mask) / (image.shape[0] * image.shape[1])
#     return white_area_per_pixel

# # Calculate the average pixel values and white areas
# real_avg_values, real_white_areas = calculate_metrics(real_images)
# synthetic_avg_values, synthetic_white_areas = calculate_metrics(synthetic_images)
# print(len(real_white_areas), len(synthetic_white_areas))

# # Normalize the frequencies using probability density estimation
# real_avg_values_normalized = norm.pdf(np.linspace(min(real_avg_values), max(real_avg_values), 100), np.mean(real_avg_values), np.std(real_avg_values))
# synthetic_avg_values_normalized = norm.pdf(np.linspace(min(synthetic_avg_values), max(synthetic_avg_values), 100), np.mean(synthetic_avg_values), np.std(synthetic_avg_values))
# real_white_areas_normalized = norm.pdf(np.linspace(min(real_white_areas), max(real_white_areas), 100), np.mean(real_white_areas), np.std(real_white_areas))
# synthetic_white_areas_normalized = norm.pdf(np.linspace(min(synthetic_white_areas), max(synthetic_white_areas), 100), np.mean(synthetic_white_areas), np.std(synthetic_white_areas))

# fontSIZE = 12


# # Gráfica de la distribución de valores promedio de píxeles sin normalizar
# plt.figure(figsize=(10, 6))
# plt.hist(real_avg_values, bins=50, alpha=0.5, label='Real', density=False)
# plt.hist(synthetic_avg_values, bins=50, alpha=0.5, label='Synthetic', density=False)
# plt.xlabel('Average Pixel Value', fontsize=fontSIZE)
# plt.ylabel('Frequency', fontsize=fontSIZE)
# plt.legend(fontsize=fontSIZE)
# plt.title('Distribution of Average Pixel Values Between Real and Synthetic Images', fontsize=fontSIZE)
# plt.savefig('results/distribution_comparison_avg_unnormalized.png')
# plt.close()

# # Gráfica de la distribución de áreas blancas sin normalizar
# plt.figure(figsize=(10, 6))
# plt.hist(real_white_areas, bins=50, alpha=0.5, label='Real', density=False)
# plt.hist(synthetic_white_areas, bins=50, alpha=0.5, label='Synthetic', density=False)
# plt.xlabel('White Area (pixels)', fontsize=fontSIZE)
# plt.ylabel('Frequency', fontsize=fontSIZE)
# plt.legend(fontsize=fontSIZE)
# plt.title('Distribution of White Areas Between Real and Synthetic Images', fontsize=fontSIZE)
# plt.savefig('results/distribution_comparison_white_area_unnormalized.png')
# plt.close()



# # Plot the distributions of average pixel values
# plt.figure(figsize=(10, 6))
# plt.plot(np.linspace(min(real_avg_values), max(real_avg_values), 100), real_avg_values_normalized, label='Real', alpha=0.5)
# plt.plot(np.linspace(min(synthetic_avg_values), max(synthetic_avg_values), 100), synthetic_avg_values_normalized, label='Synthetic', alpha=0.5)
# plt.xlabel('Average Pixel Value', fontsize=fontSIZE)
# plt.ylabel('Normalized Frequency', fontsize=fontSIZE)
# plt.legend(fontsize=fontSIZE)
# plt.xticks(fontsize=fontSIZE)
# plt.yticks(fontsize=fontSIZE)
# plt.title('Normalized Distribution of Average Pixel Values Between Real and Synthetic Images', fontsize=fontSIZE)

# # Save the figure
# plt.savefig('results/distribution_comparison_avg_normalized.png')
# plt.close()  # Close the plot explicitly if you don't want it to display in an interactive notebook/session

# # Plot the distributions of white areas
# plt.figure(figsize=(10, 6))
# plt.plot(np.linspace(min(real_white_areas), max(real_white_areas), 100), real_white_areas_normalized, label='Real', alpha=0.5)
# plt.plot(np.linspace(min(synthetic_white_areas), max(synthetic_white_areas), 100), synthetic_white_areas_normalized, label='Synthetic', alpha=0.5)
# plt.xlabel('Normalized Wound Area (pixels)', fontsize=fontSIZE)
# plt.ylabel('Normalized Frequency', fontsize=fontSIZE)
# plt.legend(fontsize=fontSIZE)
# plt.xticks(fontsize=fontSIZE)
# plt.yticks(fontsize=fontSIZE)
# plt.title('Normalized Distribution of Wound Areas Between Real and Synthetic Images', fontsize=fontSIZE)

# # Save the figure
# plt.savefig('results/distribution_comparison_white_area_normalized.png')
# plt.close()  # Close the plot explicitly if you don't want it to display in an interactive notebook/session


# # Calculate histograms
# real_avg_values_hist, _ = np.histogram(real_avg_values, bins=100, density=True)
# synthetic_avg_values_hist, _ = np.histogram(synthetic_avg_values, bins=100, density=True)
# real_white_areas_hist, _ = np.histogram(real_white_areas, bins=100, density=True)
# synthetic_white_areas_hist, _ = np.histogram(synthetic_white_areas, bins=100, density=True)

# # Plot the distributions using histograms
# plt.figure(figsize=(10, 6))
# plt.plot(real_avg_values_hist, label='Real', alpha=0.5)
# plt.plot(synthetic_avg_values_hist, label='Synthetic', alpha=0.5)
# plt.xlabel('Bin', fontsize=fontSIZE)
# plt.ylabel('Normalized Frequency', fontsize=fontSIZE)
# plt.legend(fontsize=fontSIZE)
# plt.xticks(fontsize=fontSIZE)
# plt.yticks(fontsize=fontSIZE)
# plt.title('Normalized Distribution of Average Pixel Values Between Real and Synthetic Images', fontsize=fontSIZE)

# # Save the figure
# plt.savefig('results/distribution_comparison_avg_histogram.png')
# plt.close()

# # Plot the distributions of white areas using histograms
# plt.figure(figsize=(10, 6))
# plt.plot(real_white_areas_hist, label='Real', alpha=0.5)
# plt.plot(synthetic_white_areas_hist, label='Synthetic', alpha=0.5)
# plt.xlabel('Bin', fontsize=fontSIZE)
# plt.ylabel('Normalized Frequency', fontsize=fontSIZE)
# plt.legend(fontsize=fontSIZE)
# plt.xticks(fontsize=fontSIZE)
# plt.yticks(fontsize=fontSIZE)
# plt.title('Normalized Distribution of White Areas Between Real and Synthetic Images', fontsize=fontSIZE)

# # Save the figure
# plt.savefig('results/distribution_comparison_white_area_histogram.png')
# plt.close()



# def extract_time_series(white_areas, num_time_stamps=7):
#     time_series = []
#     for i in range(0, len(white_areas), num_time_stamps):
#         time_series.append(white_areas[i:i + num_time_stamps])
#     return time_series
# # Extract time-series for real and synthetic images
# real_time_stamps = extract_time_series(real_white_areas)
# synthetic_time_stamps = extract_time_series(synthetic_white_areas)

# # Calculate the total white area for each time-stamp
# real_areas = [sum(time_stamp) for time_stamp in real_time_stamps]
# synthetic_areas = [sum(time_stamp) for time_stamp in synthetic_time_stamps]
# print("real_areas", len(real_areas), "synthetic_areas",len(synthetic_areas))




# # Define el tamaño de fuente
# font_size = 12

# # Asegúrate de que real_areas y synthetic_areas tengan la misma longitud
# max_len = max(len(real_areas), len(synthetic_areas))
# real_areas.extend([0] * (max_len - len(real_areas)))
# synthetic_areas.extend([0] * (max_len - len(synthetic_areas)))

# # Divide las áreas en grupos de 7 time-stamps y normaliza cada grupo para que sume 1
# real_areas_grouped = [real_areas[i:i+7] for i in range(0, max_len, 7)]
# real_areas_normalized = [group / np.sum(group) for group in real_areas_grouped]

# synthetic_areas_grouped = [synthetic_areas[i:i+7] for i in range(0, max_len, 7)]
# synthetic_areas_normalized = [group / np.sum(group) for group in synthetic_areas_grouped]

# # Etiquetas de los grupos de 7 time-stamps
# time_stamps = np.arange(1, len(real_areas_normalized) + 1)

# # Gráfica de barras para mostrar la distribución normalizada en grupos de 7 time-stamps
# plt.figure(figsize=(10, 6))
# width = 0.4  # Ancho de las barras

# plt.bar(time_stamps - width/2, real_areas_normalized, width=width, label='Real', alpha=0.5)
# plt.bar(time_stamps + width/2, synthetic_areas_normalized, width=width, label='Synthetic', alpha=0.5)
# plt.xlabel('Group of 7 Time-stamps')
# plt.ylabel('Normalized Area')
# plt.title('Normalized White Area Distribution in Groups of 7 Time-stamps')
# plt.legend()
# plt.grid(True)

# # Guarda la gráfica
# plt.savefig('results/white_area_normalized_distribution.png')
# plt.show()



#!venv/bin/python3

import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.stats import norm

# Function to get dimensions of a reference real image
def get_real_image_dimensions(reference_image_path):
    img = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        return img.shape
    return None


CELL_TYPE = 1 #0:monolayer; 1:sphere


# Define paths to the directories
real_images_base_dir = 'data/real_segmentations/Monolayer/' if CELL_TYPE == 0 else 'data/real_segmentations/Sphere/'
synthetic_images_base_dir = 'data/synth_monolayer_old/' if CELL_TYPE == 0 else 'data/synth_spheres_old/'
TIME_STEPS = 7 if CELL_TYPE == 0 else 6

# Reference path to a real image
reference_real_image_path = 'data/real_segmentations/Monolayer/Monolayer_1/gray/Monolayer_1_type_0_segmentation_1.png' if CELL_TYPE == 0 else 'data/real_segmentations/Sphere/Sphere_1/gray/Sphere_1_type_1_segmentation_1.png'
reference_dimensions = get_real_image_dimensions(reference_real_image_path)





def extract_time_series(white_areas, num_time_stamps=TIME_STEPS):
    time_series = []
    for i in range(0, len(white_areas), num_time_stamps):
        time_series.append(white_areas[i:i + num_time_stamps])
    return time_series


# Function to recursively load images from a directory and its subdirectories
def load_images_from_directory(base_directory, real=False, num_time_stamps=TIME_STEPS):
    images = []
    for root, dirs, files in os.walk(base_directory):
        if real:
            # Check if it's a real image directory and enter the "gray" folder
            if "gray" in root:
                time_stamp_images = []
                for file in files:
                    if file.endswith(".png"):
                        img_path = os.path.join(root, file)
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            time_stamp_images.append(img)
                if len(time_stamp_images) > num_time_stamps:
                    time_stamp_images = time_stamp_images[:num_time_stamps]
                images.extend(time_stamp_images)
        else:
            time_stamp_images = []
            
            files = files[1:]  #skip first file
            # files = files[:-1] #skip last file

            for idx, file in enumerate(files):
                if file.endswith(".png"):
                    print("idx", idx)
                    img_path = os.path.join(root, file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        # Resize synthetic images
                        # if reference_dimensions:
                        #     img = cv2.resize(img, reference_dimensions[::-1])
                        time_stamp_images.append(img)

                    if len(time_stamp_images) > num_time_stamps or idx == len(files) - 1:
                        time_stamp_images = time_stamp_images[:num_time_stamps]
                        images.extend(time_stamp_images)
                        time_stamp_images = []
    return images


# Load the real and synthetic images
real_images = load_images_from_directory(real_images_base_dir, real=True)
synthetic_images = load_images_from_directory(synthetic_images_base_dir, real=False)

print("Real shape:", len(real_images), real_images[0].shape)
print("Synthetic shape:", len(synthetic_images), synthetic_images[0].shape)


# Function to calculate the average pixel values and white area of images
def calculate_metrics(images):
    avg_values = [np.mean(image) for image in images]
    white_areas = [calculate_white_area(image) for image in images]
    return avg_values, white_areas

# Function to calculate the white area in an image
def calculate_white_area(image):
    # _, binary_mask = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
    # white_area = cv2.countNonZero(binary_mask)
    _, binary_mask = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
    white_area_per_pixel = cv2.countNonZero(binary_mask) / (image.shape[0] * image.shape[1])
    return white_area_per_pixel

# Calculate the average pixel values and white areas
real_avg_values, real_white_areas = calculate_metrics(real_images)
synthetic_avg_values, synthetic_white_areas = calculate_metrics(synthetic_images)
print(len(real_white_areas), len(synthetic_white_areas))

# Normalize the frequencies using probability density estimation
real_avg_values_normalized = norm.pdf(np.linspace(min(real_avg_values), max(real_avg_values), 100), np.mean(real_avg_values), np.std(real_avg_values))
synthetic_avg_values_normalized = norm.pdf(np.linspace(min(synthetic_avg_values), max(synthetic_avg_values), 100), np.mean(synthetic_avg_values), np.std(synthetic_avg_values))
real_white_areas_normalized = norm.pdf(np.linspace(min(real_white_areas), max(real_white_areas), 100), np.mean(real_white_areas), np.std(real_white_areas))
synthetic_white_areas_normalized = norm.pdf(np.linspace(min(synthetic_white_areas), max(synthetic_white_areas), 100), np.mean(synthetic_white_areas), np.std(synthetic_white_areas))

fontSIZE = 15


# Gráfica de la distribución de valores promedio de píxeles sin normalizar
plt.figure(figsize=(10, 6))
plt.hist(real_avg_values, bins=50, alpha=0.5, label='Real', density=False)
plt.hist(synthetic_avg_values, bins=50, alpha=0.5, label='Synthetic', density=False)
plt.xlabel('Average Pixel Value', fontsize=fontSIZE)
plt.ylabel('Frequency', fontsize=fontSIZE)
plt.legend(fontsize=fontSIZE)
plt.title('Distribution of Average Pixel Values Between Real and Synthetic Images', fontsize=fontSIZE)
plt.savefig('results/distribution_comparison_avg_unnormalized.png')
plt.close()

# Gráfica de la distribución de áreas blancas sin normalizar
plt.figure(figsize=(10, 6))
plt.hist(real_white_areas, bins=50, alpha=0.5, label='Real', density=False)
plt.hist(synthetic_white_areas, bins=50, alpha=0.5, label='Synthetic', density=False)
plt.xlabel('Wound Area (pixels)', fontsize=fontSIZE)
plt.ylabel('Frequency', fontsize=fontSIZE)
plt.legend(fontsize=fontSIZE)
plt.title('Distribution of Wound Areas Between Real and Synthetic Images', fontsize=fontSIZE)
plt.savefig('results/distribution_comparison_white_area_unnormalized.png')
plt.close()



# Plot the distributions of average pixel values
plt.figure(figsize=(10, 6))
plt.plot(np.linspace(min(real_avg_values), max(real_avg_values), 100), real_avg_values_normalized, label='Real', alpha=0.5)
plt.plot(np.linspace(min(synthetic_avg_values), max(synthetic_avg_values), 100), synthetic_avg_values_normalized, label='Synthetic', alpha=0.5)
plt.xlabel('Average Pixel Value', fontsize=fontSIZE)
plt.ylabel('Normalized Frequency', fontsize=fontSIZE)
plt.legend(fontsize=fontSIZE)
plt.xticks(fontsize=fontSIZE)
plt.yticks(fontsize=fontSIZE)
plt.title('Normalized Distribution of Average Pixel Values Between Real and Synthetic Images', fontsize=fontSIZE)

# Save the figure
plt.savefig('results/distribution_comparison_avg_normalized.png')
plt.close()  # Close the plot explicitly if you don't want it to display in an interactive notebook/session

# Plot the distributions of Wound areas
plt.figure(figsize=(10, 6))
plt.plot(np.linspace(min(real_white_areas), max(real_white_areas), 100), real_white_areas_normalized, label='Real', alpha=0.5)
plt.plot(np.linspace(min(synthetic_white_areas), max(synthetic_white_areas), 100), synthetic_white_areas_normalized, label='Synthetic', alpha=0.5)
plt.xlabel('Normalized Wound Area (pixels)', fontsize=fontSIZE)
plt.ylabel('Normalized Frequency', fontsize=fontSIZE)
plt.legend(fontsize=fontSIZE)
plt.xticks(fontsize=fontSIZE)
plt.yticks(fontsize=fontSIZE)
plt.title('Normalized Distribution of Wound Areas Between Real and Synthetic Images', fontsize=fontSIZE)

# Save the figure
plt.savefig('results/distribution_comparison_white_area_normalized.png')
plt.close()  # Close the plot explicitly if you don't want it to display in an interactive notebook/session


# Calculate histograms
real_avg_values_hist, _ = np.histogram(real_avg_values, bins=100, density=True)
synthetic_avg_values_hist, _ = np.histogram(synthetic_avg_values, bins=100, density=True)
real_white_areas_hist, _ = np.histogram(real_white_areas, bins=100, density=True)
synthetic_white_areas_hist, _ = np.histogram(synthetic_white_areas, bins=100, density=True)

# Plot the distributions using histograms
plt.figure(figsize=(10, 6))
plt.plot(real_avg_values_hist, label='Real', alpha=0.5)
plt.plot(synthetic_avg_values_hist, label='Synthetic', alpha=0.5)
plt.xlabel('Bin', fontsize=fontSIZE)
plt.ylabel('Normalized Frequency', fontsize=fontSIZE)
plt.legend(fontsize=fontSIZE)
plt.xticks(fontsize=fontSIZE)
plt.yticks(fontsize=fontSIZE)
plt.title('Normalized Distribution of Average Pixel Values Between Real and Synthetic Images', fontsize=fontSIZE)

# Save the figure
plt.savefig('results/distribution_comparison_avg_histogram.png')
plt.close()

# Plot the distributions of white areas using histograms
plt.figure(figsize=(10, 6))
plt.plot(real_white_areas_hist, label='Real', alpha=0.5)
plt.plot(synthetic_white_areas_hist, label='Synthetic', alpha=0.5)
plt.xlabel('Bin', fontsize=fontSIZE)
plt.ylabel('Normalized Frequency', fontsize=fontSIZE)
plt.legend(fontsize=fontSIZE)
plt.xticks(fontsize=fontSIZE)
plt.yticks(fontsize=fontSIZE)
plt.title('Normalized Distribution of Wound Areas Between Real and Synthetic Images', fontsize=fontSIZE)

# Save the figure
plt.savefig('results/distribution_comparison_white_area_histogram.png')
plt.close()



def extract_time_series(white_areas, num_time_stamps=TIME_STEPS):
    time_series = []
    for i in range(0, len(white_areas), num_time_stamps):
        time_series.append(white_areas[i:i + num_time_stamps])
    return time_series
# Extract time-series for real and synthetic images
real_time_stamps = extract_time_series(real_white_areas)
synthetic_time_stamps = extract_time_series(synthetic_white_areas)

# Calculate the total white area for each time-stamp
real_areas = [sum(time_stamp) for time_stamp in real_time_stamps]
synthetic_areas = [sum(time_stamp) for time_stamp in synthetic_time_stamps]
print("real_areas", len(real_areas), "synthetic_areas",len(synthetic_areas))




# Define el tamaño de fuente
font_size = 12

# Asegúrate de que real_areas y synthetic_areas tengan la misma longitud
max_len = max(len(real_areas), len(synthetic_areas))
real_areas.extend([0] * (max_len - len(real_areas)))
synthetic_areas.extend([0] * (max_len - len(synthetic_areas)))

# Divide las áreas en grupos de 7 time-stamps y normaliza cada grupo para que sume 1
real_areas_grouped = [real_areas[i:i+7] for i in range(0, max_len, 7)]
# real_areas_normalized = [group / np.sum(group) for group in real_areas_grouped]

synthetic_areas_grouped = [synthetic_areas[i:i+7] for i in range(0, max_len, 7)]
# synthetic_areas_normalized = [group / np.sum(group) for group in synthetic_areas_grouped]

# Ensure both have the same length
max_len = max(len(real_areas_grouped), len(synthetic_areas_grouped))
real_areas_grouped.extend([[]] * (max_len - len(real_areas_grouped)))
synthetic_areas_grouped.extend([[]] * (max_len - len(synthetic_areas_grouped)))


def normalize_groups(groups):
    """Normalize groups so each sums to 1, handling empty or zero cases gracefully."""
    normalized = []
    for group in groups:
        total = np.sum(group)
        if total == 0:
            normalized.append(group)  # Keep as is if total is zero
        else:
            normalized.append(group / total)  # Normalize otherwise
    return normalized
# Normalize areas
real_areas_normalized = normalize_groups(real_areas_grouped)
synthetic_areas_normalized = normalize_groups(synthetic_areas_grouped)


# Etiquetas de los grupos de 7 time-stamps
time_stamps = np.arange(1, len(real_areas_normalized) + 1)

# Gráfica de barras para mostrar la distribución normalizada en grupos de 7 time-stamps
width = 0.4  # Ancho de las barras
plt.figure(figsize=(12, 6))

# Plot real and synthetic data as lines
plt.plot(np.arange(len(real_areas)), real_areas, label='Real', marker='o', linestyle='-', color='blue')
plt.plot(np.arange(len(synthetic_areas)), synthetic_areas, label='Synthetic', marker='x', linestyle='-', color='orange')

plt.xlabel('Time Points')
plt.ylabel('Wound Area')
plt.title('Comparison of Wound Area Distribution')
plt.legend()
plt.grid(True)

plt.savefig('results/comparison_white_area_line.png')
plt.show()

plt.figure(figsize=(12, 6))

plt.scatter(np.arange(len(real_areas)), real_areas, label='Real', color='blue')
plt.scatter(np.arange(len(synthetic_areas)), synthetic_areas, label='Synthetic', color='orange')

plt.xlabel('Time Points')
plt.ylabel('Wound Area')
plt.title('Comparison of Wound Area Distribution')
plt.legend()
plt.grid(True)

plt.savefig('results/comparison_white_area_scatter.png')
plt.show()




import pandas as pd

def create_comparison_table(real_metrics, synthetic_metrics):
    """Create a comparison table for metrics between real and synthetic data."""
    comparison_data = {
        'Metric': ['Mean Wound Area', 'Std. Dev Wound Area'],
        'Real': [np.mean(real_metrics), np.std(real_metrics)],
        'Synthetic': [np.mean(synthetic_metrics), np.std(synthetic_metrics)]
    }

    return pd.DataFrame(comparison_data)

# Generate and print the comparison table
comparison_table = create_comparison_table(real_white_areas, synthetic_white_areas)
print(comparison_table)









# ********************************
# Step-by-Step Comparison: Instead of comparing each time series as a whole, compare each time step within each series. This would allow you to see how healing progresses over time and how similar or different synthetic and real data are at each stage.
# Averaging by Time Step: Group the data by time step, then compute the mean and standard deviation for each time step across all real and synthetic series. This provides a clear indication of how the healing patterns compare on average.
# Statistical Tests: You can also perform statistical tests, like a paired t-test or ANOVA, to quantify the difference between real and synthetic data at each time step.
# ********************************


import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.stats import ttest_ind, norm

# Second function to load images from a directory and organize them by series
def load_images(base_directory, num_time_stamps=TIME_STEPS, synthetic=False):
    series = []
    for root, dirs, _ in os.walk(base_directory):
        files = []  # Inicializamos la variable antes del bloque condicional
        time_stamp_images = []  # Inicializamos la variable antes de entrar al bucle

        if not synthetic:
            if "gray" in dirs:
                files = os.listdir(os.path.join(root, "gray"))
                time_stamp_images = [cv2.imread(os.path.join(root, "gray", file), cv2.IMREAD_GRAYSCALE) for file in files if file.endswith(".png")]
        else:
            
            files = files[1:]  #skip first file
            # files = files[:-1] #skip last file

            files = [file for file in os.listdir(root) if file.endswith(".png")]
            time_stamp_images = [cv2.imread(os.path.join(root, file), cv2.IMREAD_GRAYSCALE) for file in files]

        if len(time_stamp_images) > num_time_stamps:
            time_stamp_images = time_stamp_images[:num_time_stamps]

        if time_stamp_images:
            #Resize synthetic images
            if synthetic and reference_dimensions:
                time_stamp_images = [cv2.resize(img, reference_dimensions[::-1]) for img in time_stamp_images if img is not None]

            series.append(time_stamp_images)

    return series



def calculate_white_area(image):
    _, binary_mask = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
    return cv2.countNonZero(binary_mask) / (image.shape[0] * image.shape[1])

def calculate_metrics(series):
    return [[calculate_white_area(image) for image in series_set] for series_set in series]

# Load datasets
real_series = load_images(real_images_base_dir)
synthetic_series = load_images(synthetic_images_base_dir, synthetic=True)

real_metrics = calculate_metrics(real_series)
synthetic_metrics = calculate_metrics(synthetic_series)

# Prepare data for comparison
from scipy.stats import wilcoxon, pearsonr
def prepare_data_for_comparison(real_metrics, synthetic_metrics):
    real_means = np.mean(real_metrics, axis=0)
    real_stds = np.std(real_metrics, axis=0)

    synthetic_means = np.mean(synthetic_metrics, axis=0)
    synthetic_stds = np.std(synthetic_metrics, axis=0)

    # Paired Wilcoxon test for each time step
    p_values = [wilcoxon(real_metrics[i], synthetic_metrics[i])[1] for i in range(len(real_means))]

    # Correlation coefficient between the overall trends
    corr_coeff, corr_p_value = pearsonr(real_means, synthetic_means)

    return real_means, real_stds, synthetic_means, synthetic_stds, p_values, corr_coeff, corr_p_value
real_means, real_stds, synthetic_means, synthetic_stds, p_values, corr_coeff, corr_p_value = prepare_data_for_comparison(real_metrics, synthetic_metrics)

# Plot comparison
time_steps = np.arange(1, len(real_means) + 1)

plt.figure(figsize=(8, 6))

plt.errorbar(time_steps, real_means, yerr=real_stds, label='Real', capsize=3, fmt='-o')
plt.errorbar(time_steps, synthetic_means, yerr=synthetic_stds, label='Synthetic', capsize=3, fmt='-x')

plt.xlabel('Time Steps')
plt.ylabel('Mean Wound Area')
plt.title('Comparison of Wound Area by Time Step')
plt.legend()
plt.grid(True)

plt.savefig('results/white_area_comparison_by_time_step.png', bbox_inches='tight', pad_inches=0.1)
plt.show()

# # Display p-values in a table format
# comparison_data = {
#     'Time Step': time_steps,
#     'P-Value': p_values,
# }

# comparison_table = pd.DataFrame(comparison_data)
# print(comparison_table)


# Display table with p-values and other statistics
comparison_data = {
    'Time Step': time_steps,
    'P-Value': p_values,
    'Real Mean': real_means,
    'Real Std': real_stds,
    'Synthetic Mean': synthetic_means,
    'Synthetic Std': synthetic_stds,
}

comparison_table = pd.DataFrame(comparison_data)

print(f"Correlation between Real and Synthetic Trends: {corr_coeff:.4f} (p-value: {corr_p_value:.4f})\n")

print(comparison_table)

# Include information about statistical significance
comparison_table['Significant'] = comparison_table['P-Value'].apply(lambda p: "Yes" if p < 0.05 else "No")

print("\nSummary with Significance:\n")
print(comparison_table)

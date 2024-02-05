#!venv/bin/python3

import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.stats import norm

# Define paths to the directories
real_images_base_dir = 'data/real_segmentations/Monolayer/'
synthetic_images_base_dir = 'data/synth_monolayer_old/'

# Function to recursively load images from a directory and its subdirectories
def load_images_from_directory(base_directory, real=False, num_time_stamps=7):
    images = []
    for root, dirs, files in os.walk(base_directory):
        if real:
            # Check if it's a real image directory and enter the "gray" folder
            if "gray" in root:
                time_stamp_images = []  # Store images for each time-stamp
                for file in files:
                    if file.endswith(".png"):
                        img_path = os.path.join(root, file)
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            time_stamp_images.append(img)
                # Check if we have more than 7 images for this time-stamp
                if len(time_stamp_images) > num_time_stamps:
                    time_stamp_images = time_stamp_images[:num_time_stamps]  # Keep only the first 7 images
                images.extend(time_stamp_images)
        else:
            # For synthetic images, check the number of time-stamps
            time_stamp_images = []  # Store images for each time-stamp
            for file in files:
                if file.endswith(".png"):
                    img_path = os.path.join(root, file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        time_stamp_images.append(img)
                    # Check if we have more than 7 images or if we have reached the end of the time-stamp
                    if len(time_stamp_images) > num_time_stamps or (files.index(file) == len(files) - 1 and len(time_stamp_images) > 0):
                        time_stamp_images = time_stamp_images[:num_time_stamps]  # Keep only the first 7 images
                        images.extend(time_stamp_images)
                        time_stamp_images = []  # Reset for the next time-stamp
    return images



# Load the real and synthetic images
real_images = load_images_from_directory(real_images_base_dir, real=True)
synthetic_images = load_images_from_directory(synthetic_images_base_dir)

print(len(real_images), real_images[0].shape)
print(len(synthetic_images), synthetic_images[0].shape)


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

fontSIZE = 12
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

# Plot the distributions of white areas
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
plt.title('Normalized Distribution of White Areas Between Real and Synthetic Images', fontsize=fontSIZE)

# Save the figure
plt.savefig('results/distribution_comparison_white_area_histogram.png')
plt.close()



def extract_time_series(white_areas, num_time_stamps=7):
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
real_areas_normalized = [group / np.sum(group) for group in real_areas_grouped]

synthetic_areas_grouped = [synthetic_areas[i:i+7] for i in range(0, max_len, 7)]
synthetic_areas_normalized = [group / np.sum(group) for group in synthetic_areas_grouped]

# Etiquetas de los grupos de 7 time-stamps
time_stamps = np.arange(1, len(real_areas_normalized) + 1)

# Gráfica de barras para mostrar la distribución normalizada en grupos de 7 time-stamps
plt.figure(figsize=(10, 6))
width = 0.4  # Ancho de las barras

plt.bar(time_stamps - width/2, real_areas_normalized, width=width, label='Real', alpha=0.5)
plt.bar(time_stamps + width/2, synthetic_areas_normalized, width=width, label='Synthetic', alpha=0.5)
plt.xlabel('Group of 7 Time-stamps')
plt.ylabel('Normalized Area')
plt.title('Normalized White Area Distribution in Groups of 7 Time-stamps')
plt.legend()
plt.grid(True)

# Guarda la gráfica
plt.savefig('results/white_area_normalized_distribution.png')
plt.show()
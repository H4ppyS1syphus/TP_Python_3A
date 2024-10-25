# utils.py

import os
import psutil
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.morphology import opening, closing, ball, remove_small_objects
from skimage.measure import marching_cubes
from scipy.ndimage import uniform_filter
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
import tifffile as tiff
import pyvista as pv
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import logging

from config import (
    MASKS_DIR,
    MASK_FILENAMES,
    ORIGINAL_MRI_PATH,
    DENOISED_MRI_PATH,
    DENOISED_TIFF_PATH,
    OUTPUT_HTML,
    OUTPUT_SLICES_DIR,
    K,
    BATCH_SIZE,
    STRUCTURING_ELEMENT_RADIUS,
    MIN_SIZE
)

# ----- Utility Functions -----
def print_memory_usage():
    """
    Prints the current memory usage of the process in MB.
    """
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 2)  # Convert bytes to MB
    print(f"Current memory usage: {mem:.2f} MB")

def setup_logging(enable_log, log_file):
    """
    Sets up logging configuration.
    """
    if enable_log:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logging.basicConfig(
            filename=log_file,
            filemode='a',
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )
        logging.info("Logging initiated.")
    else:
        logging.basicConfig(level=logging.CRITICAL)  # Suppress logs if not enabled

def log_memory_usage():
    """
    Logs the current memory usage.
    """
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 2)  # MB
    logging.info(f"Current memory usage: {mem:.2f} MB")

def load_image(image_path, dtype=np.float32):
    """
    Loads an image and converts it to float.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file '{image_path}' does not exist.")
    print(f"Loading image from '{image_path}'...")
    logging.info(f"Loading image from '{image_path}'.")
    img = io.imread(image_path)
    img = img_as_float(img).astype(dtype)
    print(f"Image loaded with shape {img.shape}.")
    logging.info(f"Image loaded with shape {img.shape}.")
    return img

def save_image(image, image_path):
    """
    Saves an image to a TIFF file.
    """
    print(f"Saving image to '{image_path}'...")
    logging.info(f"Saving image to '{image_path}'.")
    image_uint16 = (image * 65535).astype(np.uint16)
    tiff.imwrite(image_path, image_uint16, imagej=True)
    print("Image saved.")
    logging.info("Image saved.")

def denoise_image(original_img, denoised_tiff_path):
    """
    Denoises the original image using Non-local Means and saves the denoised image.
    """
    if os.path.exists(denoised_tiff_path):
        print("Loading denoised image...")
        denoised_img = io.imread(denoised_tiff_path).astype(np.float32) / 65535.0
        print("Denoised image loaded.")
        logging.info("Denoised image loaded.")
    else:
        print("Denoised image not found. Performing denoising...")
        logging.info("Denoised image not found. Performing denoising.")
        
        num_slices = original_img.shape[0]
        denoised_img = np.zeros_like(original_img)
        
        for i in tqdm(range(num_slices), desc="Denoising slices"):
            img_slice = original_img[i, :, :]
            
            # Estimate the noise standard deviation
            sigma_est = np.mean(estimate_sigma(img_slice, channel_axis=None))
            logging.info(f"Slice {i}: Estimated noise sigma = {sigma_est}")
            
            # Define patch settings for Non-local Means denoising
            patch_kw = dict(
                patch_size=5, patch_distance=6, channel_axis=None  # 5x5 patches, 6x6 search area
            )
            
            # Apply Non-local Means denoising
            denoised_slice = denoise_nl_means(
                img_slice,
                h=0.8 * sigma_est,
                fast_mode=True,
                **patch_kw
            )
            denoised_img[i, :, :] = denoised_slice
            logging.info(f"Slice {i}: Denoising complete.")
        
        # Save the denoised image
        save_image(denoised_img, denoised_tiff_path)
    
    print_memory_usage()
    log_memory_usage()
    return denoised_img

def normalize_intensity(denoised_img):
    """
    Normalizes the intensity of the denoised image using standard scaling.
    """
    print("Starting intensity normalization...")
    logging.info("Starting intensity normalization.")
    
    num_slices, height, width = denoised_img.shape
    intensities = denoised_img.flatten().reshape(-1, 1)
    
    scaler = StandardScaler()
    intensities_scaled = scaler.fit_transform(intensities)
    denoised_img_normalized = intensities_scaled.reshape(denoised_img.shape)
    
    print("Intensity normalization complete.")
    logging.info("Intensity normalization complete.")
    print_memory_usage()
    log_memory_usage()
    return denoised_img_normalized

def plot_slices(denoised_img_normalized, n_slices, output_dir):
    """
    Saves n slices with matplotlib after K-Means clustering.
    """
    os.makedirs(output_dir, exist_ok=True)
    total_slices = denoised_img_normalized.shape[0]
    slice_indices = np.linspace(0, total_slices - 1, n_slices, dtype=int)
    
    for idx in slice_indices:
        plt.figure(figsize=(6, 6))
        plt.imshow(denoised_img_normalized[idx], cmap='gray')
        plt.title(f'Normalized Denoised MRI Slice {idx}')
        plt.axis('off')
        slice_path = os.path.join(output_dir, f'slice_{idx}.png')
        plt.savefig(slice_path, bbox_inches='tight')
        plt.close()
        print(f"Saved slice {idx} as '{slice_path}'.")
        logging.info(f"Saved slice {idx} as '{slice_path}'.")
    
    print(f"All {n_slices} slices saved to '{output_dir}'.")
    logging.info(f"All {n_slices} slices saved to '{output_dir}'.")

def compute_neighborhood_statistics(denoised_img_normalized, neighborhood_size=3):
    """
    Computes neighborhood mean and variance for each voxel.
    """
    print("Computing neighborhood mean...")
    logging.info("Computing neighborhood mean.")
    neighborhood_mean = uniform_filter(denoised_img_normalized, size=neighborhood_size, mode='reflect')
    
    print("Computing neighborhood mean of squares...")
    logging.info("Computing neighborhood mean of squares.")
    neighborhood_mean_sq = uniform_filter(denoised_img_normalized**2, size=neighborhood_size, mode='reflect')
    
    print("Computing neighborhood variance...")
    logging.info("Computing neighborhood variance.")
    neighborhood_variance = neighborhood_mean_sq - neighborhood_mean**2
    
    print("Neighborhood statistics computed.")
    logging.info("Neighborhood statistics computed.")
    print_memory_usage()
    log_memory_usage()
    return neighborhood_mean, neighborhood_variance

def visualize_feature_distributions(features, sample_size=100000):
    """
    Plots feature distributions and pairwise relationships.
    """
    print("Plotting feature distributions...")
    logging.info("Plotting feature distributions.")
    plt.figure(figsize=(18, 4))
    
    # Plot neighborhood mean distribution
    plt.subplot(1, 3, 1)
    plt.hist(features[:, 1], bins=100, color='purple')
    plt.title('Neighborhood Mean Distribution')
    plt.xlabel('Mean Intensity')
    plt.ylabel('Frequency')
    
    # Plot neighborhood variance distribution
    plt.subplot(1, 3, 2)
    plt.hist(features[:, 2], bins=100, color='orange')
    plt.title('Neighborhood Variance Distribution')
    plt.xlabel('Variance')
    plt.ylabel('Frequency')
    
    # Plot a scatter of mean vs variance
    plt.subplot(1, 3, 3)
    if features.shape[0] > sample_size:
        indices = np.random.choice(features.shape[0], size=sample_size, replace=False)
        plt.scatter(features[indices, 1], features[indices, 2], alpha=0.1, s=1)
    else:
        plt.scatter(features[:, 1], features[:, 2], alpha=0.1, s=1)
    plt.title('Mean vs Variance')
    plt.xlabel('Mean Intensity')
    plt.ylabel('Variance')
    
    plt.tight_layout()
    plt.show()
    logging.info("Feature distributions plotted.")

def analyze_feature_distributions(features, sample_size=1000):
    """
    Further analyzes feature distributions using seaborn pairplot.
    """
    print("Analyzing enhanced feature distributions with a sampled subset...")
    logging.info("Analyzing enhanced feature distributions with a sampled subset.")
    
    if features.shape[0] > sample_size:
        indices = np.random.choice(features.shape[0], size=sample_size, replace=False)
        features_sample = features[indices]
    else:
        features_sample = features
    
    df_features = pd.DataFrame(features_sample, columns=['Intensity', 'Neighborhood Mean', 'Neighborhood Variance'])
    
    sns.pairplot(df_features[['Intensity', 'Neighborhood Mean', 'Neighborhood Variance']], diag_kind='hist', markers='.', plot_kws={'alpha': 0.1})
    plt.suptitle('Pairwise Feature Relationships (Sampled)', y=1.02)
    plt.show()
    logging.info("Pairwise feature relationships plotted.")

def apply_morphology_3d(mask, selem, min_size=64):
    """
    Applies morphological opening and closing to a 3D mask to remove small objects
    and fill small holes.
    """
    logging.info("Applying morphological opening to remove small objects.")
    mask = opening(mask, selem)
    logging.info("Applying morphological closing to fill small holes.")
    mask = closing(mask, selem)
    logging.info(f"Removing objects smaller than {min_size} voxels.")
    mask = remove_small_objects(mask, min_size=min_size)
    return mask

def perform_kmeans(features, k=5, batch_size=10000):
    """
    Performs MiniBatch K-Means clustering on the provided features.
    """
    print("Performing MiniBatch K-means clustering...")
    logging.info(f"Performing MiniBatch K-Means clustering with k={k}, batch_size={batch_size}.")
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=batch_size)
    kmeans.fit(features)
    logging.info("Clustering complete.")
    print("Clustering complete.")
    print_memory_usage()
    log_memory_usage()
    return kmeans

def create_refined_masks(clustered_img, denoised_img_normalized, k, selem, min_size, masks_dir):
    """
    Creates refined masks for each cluster using morphological operations and saves them.
    """
    os.makedirs(masks_dir, exist_ok=True)
    refined_masks = {}
    cluster_means = []
    
    for i in range(k):
        print(f"\nProcessing Cluster {i}...")
        logging.info(f"Processing Cluster {i}.")
        
        cluster_mask = (clustered_img == i)
        voxel_count = cluster_mask.sum()
        logging.info(f"Cluster {i}: Initial voxel count = {voxel_count}.")
        print(f"  Cluster {i} mask voxel count: {voxel_count}")
        
        refined_mask = apply_morphology_3d(cluster_mask, selem, min_size)
        refined_voxel_count = refined_mask.sum()
        logging.info(f"Cluster {i}: Refined voxel count = {refined_voxel_count}.")
        print(f"  Cluster {i} refined mask voxel count: {refined_voxel_count}")
        
        if refined_voxel_count == 0:
            print(f"  Warning: Refined mask for Cluster {i} is empty. Consider adjusting morphological parameters.")
            logging.warning(f"Refined mask for Cluster {i} is empty.")
        
        # Save the refined mask
        mask_filename = f'csf_mask_class_{i}_refined_2.tif'
        mask_path = os.path.join(masks_dir, mask_filename)
        tiff.imwrite(mask_path, refined_mask.astype(np.uint8) * 255, imagej=True)
        print(f"  Refined mask for Cluster {i} saved as '{mask_filename}'.")
        logging.info(f"Refined mask for Cluster {i} saved as '{mask_filename}'.")
        
        # Store in dictionary
        refined_masks[i] = refined_mask
        
        # Compute mean intensity for the cluster
        if voxel_count > 0:
            cluster_mean = denoised_img_normalized.flatten()[clustered_img.flatten() == i].mean()
        else:
            cluster_mean = 0
        cluster_means.append(cluster_mean)
        logging.info(f"Cluster {i}: Mean Intensity = {cluster_mean}")
        print(f"Cluster {i}: Mean Intensity = {cluster_mean}")
    
    # Identify the cluster with the lowest mean intensity as CSF
    csf_cluster = np.argmin(cluster_means)
    print(f"CSF is identified as cluster {csf_cluster}.")
    logging.info(f"CSF is identified as cluster {csf_cluster}.")
    
    return refined_masks, csf_cluster

def plot_refined_masks_on_slices(refined_masks, denoised_img_normalized, slice_indices, k, output_dir):
    """
    Plots and saves refined masks overlaid on specified slices.
    
    Parameters:
    - refined_masks: Dictionary of refined mask arrays.
    - denoised_img_normalized: 3D numpy array of the normalized denoised MRI image.
    - slice_indices: List of slice indices to visualize.
    - k: Number of clusters/masks.
    - output_dir: Directory where the overlay images will be saved.
    """
    
    for slice_idx in slice_indices:
        if slice_idx < denoised_img_normalized.shape[0]:
            for cluster_idx in range(k):
                refined_mask = refined_masks.get(cluster_idx)
                if refined_mask is None:
                    logging.warning(f"No refined mask found for cluster {cluster_idx}. Skipping.")
                    continue
                
                # Check if slice index is within bounds for the mask
                if slice_idx >= refined_mask.shape[0]:
                    print(f"Slice index {slice_idx} is out of bounds for cluster {cluster_idx} mask with {refined_mask.shape[0]} slices.")
                    logging.warning(f"Slice index {slice_idx} is out of bounds for cluster {cluster_idx} mask.")
                    continue
                
                plt.figure(figsize=(6, 6))
                plt.imshow(denoised_img_normalized[slice_idx], cmap='gray')
                plt.imshow(refined_mask[slice_idx], cmap='jet', alpha=0.5)
                plt.title(f'Cluster {cluster_idx} Refined Mask Overlay on Slice {slice_idx}')
                plt.axis('off')
                
                # Define the filename and path
                slice_path = os.path.join(output_dir, f'slice_{slice_idx}_cluster_{cluster_idx}_overlay.png')
                plt.savefig(slice_path, bbox_inches='tight')
                plt.close()
                
                print(f"Saved slice {slice_idx} overlay for Cluster {cluster_idx} as '{slice_path}'.")
                logging.info(f"Saved slice {slice_idx} overlay for Cluster {cluster_idx} as '{slice_path}'.")
        else:
            print(f"Slice index {slice_idx} is out of bounds for image with {denoised_img_normalized.shape[0]} slices.")
            logging.warning(f"Slice index {slice_idx} is out of bounds for the denoised image.")


def extract_surface_mesh(mask_binary, step_size=2, mc_threshold=0.5):
    """
    Extracts a surface mesh from a binary mask using Marching Cubes.
    """
    verts, faces, normals, values = marching_cubes(mask_binary, level=mc_threshold, step_size=step_size, allow_degenerate=False)
    return verts, faces

def decimate_mesh(verts, faces, target_reduction=0.5):
    """
    Decimates the mesh to reduce the number of triangles.
    """
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int32)
    csf_mesh = pv.PolyData(verts, faces_pv)
    
    print(f"Initial mesh has {csf_mesh.n_points} points and {csf_mesh.n_faces} faces.")
    logging.info(f"Initial mesh has {csf_mesh.n_points} points and {csf_mesh.n_faces} faces.")
    print_memory_usage()
    log_memory_usage()
    
    csf_mesh_decimated = csf_mesh.decimate(target_reduction, inplace=False)
    
    print(f"Decimated mesh has {csf_mesh_decimated.n_points} points and {csf_mesh_decimated.n_faces} faces.")
    logging.info(f"Decimated mesh has {csf_mesh_decimated.n_points} points and {csf_mesh_decimated.n_faces} faces.")
    print_memory_usage()
    log_memory_usage()
    
    return csf_mesh_decimated

def visualize_and_save_html(original_img, denoised_img, refined_masks, k, output_html):
    """
    Visualizes the Original MRI, Denoised MRI, and CSF Masks using PyVista and saves as HTML.
    """
    print("\nStarting visualization with PyVista...")
    logging.info("Starting visualization with PyVista.")
    
    # Initialize PyVista Plotter
    rows = 3
    cols = 3
    plotter = pv.Plotter(shape=(rows, cols), title="MRI and CSF Masks Visualization", window_size=[1800, 900])
    
    # -----------------------------
    # 1. Original MRI Volume
    # -----------------------------
    plotter.subplot(0, 0)  # Top-left
    try:
        print("Adding Original MRI volume...")
        logging.info("Adding Original MRI volume.")
        plotter.add_volume(original_img, cmap="gray", opacity="linear", name="Original MRI")
    except TypeError as e:
        print(f"Failed to add Original MRI volume: {e}")
        logging.error(f"Failed to add Original MRI volume: {e}")
        plotter.add_volume(original_img, cmap="gray", opacity="linear", name="Original MRI")
    plotter.add_text("Original MRI", position="upper_left", font_size=10)
    plotter.show_axes()
    
    # -----------------------------
    # 2. Denoised MRI Volume
    # -----------------------------
    plotter.subplot(0, 1)  # Top-middle
    try:
        print("Adding Denoised MRI volume...")
        logging.info("Adding Denoised MRI volume.")
        plotter.add_volume(denoised_img, cmap="coolwarm", opacity="linear", name="Denoised MRI")
    except TypeError as e:
        print(f"Failed to add Denoised MRI volume: {e}")
        logging.error(f"Failed to add Denoised MRI volume: {e}")
        plotter.add_volume(denoised_img, cmap="coolwarm", opacity="linear", name="Denoised MRI")
    plotter.add_text("Denoised MRI", position="upper_left", font_size=10)
    plotter.show_axes()
    
    # -----------------------------
    # 3-7. CSF Mask Overlays
    # -----------------------------
    for cluster_idx in range(k):
        mesh = refined_masks.get(cluster_idx)
        if mesh is None:
            continue  # Skip if mask is not available
        
        print(f"Adding Mask {cluster_idx} mesh to visualization...")
        logging.info(f"Adding Mask {cluster_idx} mesh to visualization.")
        
        # Extract surface mesh
        mask_binary = (mesh > 0).astype(np.uint8)
        verts, faces = extract_surface_mesh(mask_binary)
        
        # Decimate mesh
        mesh_decimated = decimate_mesh(verts, faces)
        
        # Add mesh to plotter
        plotter.subplot((cluster_idx + 2) // cols, (cluster_idx + 2) % cols)  # Adjust subplot position
        plotter.add_mesh(mesh_decimated, color="red", opacity=0.5, show_edges=False, label=f"Mask {cluster_idx}")
        plotter.add_text(f"Mask {cluster_idx}", position="upper_left", font_size=10)
        plotter.show_axes()
    
    # -----------------------------
    # Optional: Hide unused subplots
    # -----------------------------
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in [(0, 0), (0, 1)] + [(int((idx + 2) / cols), (idx + 2) % cols) for idx in range(k)]:
                plotter.subplot(r, c)
                plotter.remove_actor('background')
                plotter.hide_axes()
    
    # -----------------------------
    # Save Visualization as HTML
    # -----------------------------
    print(f"\nExporting visualization to '{output_html}'...")
    logging.info(f"Exporting visualization to '{output_html}'.")
    plotter.export_html(output_html)
    print(f"Visualization successfully saved to '{output_html}'.")
    logging.info(f"Visualization successfully saved to '{output_html}'.")
    
    # -----------------------------
    # Close the Plotter
    # -----------------------------
    plotter.close()
    print_memory_usage()
    log_memory_usage()

def extract_and_decimate_meshes(refined_masks, denoised_img_normalized, k):
    """
    Extracts and decimates meshes for all refined masks.
    Returns a list of decimated meshes and their labels.
    """
    decimated_meshes = []
    mesh_labels = []
    
    for cluster_idx in range(k):
        mesh = refined_masks.get(cluster_idx)
        if mesh is None:
            continue  # Skip if mask is not available
        
        print(f"Processing Mesh for Cluster {cluster_idx}...")
        logging.info(f"Processing Mesh for Cluster {cluster_idx}.")
        
        # Extract surface mesh
        mask_binary = (mesh > 0).astype(np.uint8)
        verts, faces = extract_surface_mesh(mask_binary)
        
        # Decimate mesh
        mesh_decimated = decimate_mesh(verts, faces)
        
        decimated_meshes.append(mesh_decimated)
        mesh_labels.append(f"Mask {cluster_idx}")
    
    return decimated_meshes, mesh_labels

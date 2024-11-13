# utils.py

import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import seaborn as sns
import tifffile as tiff
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import uniform_filter, zoom
from scipy.signal import find_peaks
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from skimage import io, img_as_float, measure
from skimage.measure import label, regionprops, marching_cubes
from skimage.morphology import (
    binary_dilation,
    binary_erosion,
    ball,
    closing,
    opening,
    remove_small_objects
)
from skimage.restoration import denoise_nl_means, estimate_sigma
from tqdm import tqdm
import matplotlib.colors as colors
import itertools
import json
import pyvista as pv


from config import (
    MASKS_DIR,
    K,
    DENOISED_TIFF_PATH,
    ORIGINAL_MRI_PATH,
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
            sigma_est = np.mean(estimate_sigma(img_slice))
            logging.info(f"Slice {i}: Estimated noise sigma = {sigma_est}")
            
            # Define patch settings for Non-local Means denoising
            patch_kw = dict(
                patch_size=5, patch_distance=6, fast_mode=True
            )
            
            # Apply Non-local Means denoising
            denoised_slice = denoise_nl_means(
                img_slice,
                h=0.8 * sigma_est,
                **patch_kw
            )
            denoised_img[i, :, :] = denoised_slice
            logging.info(f"Slice {i}: Denoising complete.")
            
            # Explicitly delete variables to free memory
            del img_slice, denoised_slice, sigma_est

        # Delete original image if not needed further
        del original_img
        
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
    
    scaler = StandardScaler()
    denoised_img_flat = denoised_img.flatten().reshape(-1, 1)
    intensities_scaled = scaler.fit_transform(denoised_img_flat)
    denoised_img_normalized = intensities_scaled.reshape(denoised_img.shape)
    
    # Clean up
    del scaler, denoised_img_flat, intensities_scaled
    
    print("Intensity normalization complete.")
    logging.info("Intensity normalization complete.")
    print_memory_usage()
    log_memory_usage()
    return denoised_img_normalized


def compute_neighborhood_statistics(denoised_img_normalized, neighborhood_size=3):
    """
    Computes neighborhood mean and variance for each voxel.
    """
    print("Computing neighborhood mean and variance...")
    logging.info("Computing neighborhood statistics.")
    neighborhood_mean = uniform_filter(denoised_img_normalized, size=neighborhood_size, mode='reflect')
    neighborhood_mean_sq = uniform_filter(denoised_img_normalized**2, size=neighborhood_size, mode='reflect')
    neighborhood_variance = neighborhood_mean_sq - neighborhood_mean**2
    
    # Clean up
    del neighborhood_mean_sq
    
    print("Neighborhood statistics computed.")
    logging.info("Neighborhood statistics computed.")
    print_memory_usage()
    log_memory_usage()
    return neighborhood_mean, neighborhood_variance


def compute_z_value(denoised_img_normalized):
    """
    Computes the value along the z-axis for each voxel (x, y, z).
    Currently, this function duplicates the input image.
    """
    print("Computing z-axis value for each voxel...")
    logging.info("Computing z-axis value for each voxel.")
    
    # Assuming z-axis corresponds to the first dimension
    z_value = denoised_img_normalized.copy()
    
    print("Z-axis value for each voxel computed.")
    logging.info("Z-axis value for each voxel computed.")
    print_memory_usage()
    log_memory_usage()
    return z_value


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
    
    # Clean up
    plt.clf()
    plt.close('all')
    if features.shape[0] > sample_size:
        del indices


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
    
    # Clean up
    plt.clf()
    plt.close('all')
    del features_sample
    del df_features


def perform_kmeans(features, denoised_img_shape, k=5, batch_size=10000, connectivity_weight=0.1):
    """
    Performs MiniBatch K-Means clustering on the provided features, including spatial connectivity.

    Parameters:
    - features: numpy array of shape (n_samples, n_features)
    - denoised_img_shape: tuple representing the shape of the denoised image (num_slices, height, width)
    - k: number of clusters
    - batch_size: size of mini-batches for the MiniBatchKMeans algorithm
    - connectivity_weight: float, weight for the spatial connectivity features

    Returns:
    - kmeans: trained MiniBatchKMeans object
    """
    print("Performing MiniBatch K-means clustering with spatial connectivity...")
    logging.info(f"Performing MiniBatch K-Means clustering with k={k}, batch_size={batch_size}, connectivity_weight={connectivity_weight}.")

    # Get the spatial coordinates
    num_slices, height, width = denoised_img_shape
    z_coords, y_coords, x_coords = np.meshgrid(
        np.arange(num_slices),
        np.arange(height),
        np.arange(width),
        indexing='ij'
    )

    # Flatten and stack the spatial coordinates
    spatial_features = np.stack((
        z_coords.flatten(),
        y_coords.flatten(),
        x_coords.flatten()
    ), axis=1)

    # Normalize spatial features
    scaler = StandardScaler()
    spatial_features_scaled = scaler.fit_transform(spatial_features)

    # Apply connectivity weight
    spatial_features_scaled *= connectivity_weight

    # Combine original features with spatial features
    features_with_connectivity = np.hstack((features, spatial_features_scaled))

    # Clean up
    del spatial_features, spatial_features_scaled, z_coords, y_coords, x_coords, scaler

    logging.info(f"Features shape after adding spatial connectivity: {features_with_connectivity.shape}")
    print(f"Features shape after adding spatial connectivity: {features_with_connectivity.shape}")

    # Perform K-Means clustering
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=batch_size)
    kmeans.fit(features_with_connectivity)
    logging.info("Clustering complete with spatial connectivity.")
    print("Clustering complete with spatial connectivity.")
    print_memory_usage()
    log_memory_usage()

    # Clean up
    del features_with_connectivity
    return kmeans


def create_refined_masks(clustered_img, denoised_img_normalized, k, selem, min_size, masks_dir):
    """
    Creates refined masks for each cluster using morphological operations and saves them.

    Parameters:
    - clustered_img: 3D numpy array of cluster labels
    - denoised_img_normalized: 3D numpy array of the normalized denoised MRI image
    - k: number of clusters
    - selem: structuring element for morphological operations
    - min_size: minimum size for removing small objects
    - masks_dir: directory to save refined masks

    Returns:
    - refined_masks: Dictionary mapping cluster index to refined mask
    - csf_cluster: Cluster index identified as CSF based on lowest mean intensity
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

        # Apply morphological operations
        refined_mask = opening(cluster_mask, selem)
        refined_mask = closing(refined_mask, selem)
        refined_mask = remove_small_objects(refined_mask, min_size=min_size)
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
    csf_cluster = np.argmax(cluster_means)
    print(f"CSF is identified as cluster {csf_cluster}.")
    logging.info(f"CSF is identified as cluster {csf_cluster}.")

    # Clean up
    del cluster_means
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

                # Clean up
                plt.clf()
                plt.close('all')
        else:
            print(f"Slice index {slice_idx} is out of bounds for image with {denoised_img_normalized.shape[0]} slices.")
            logging.warning(f"Slice index {slice_idx} is out of bounds for the denoised image.")


def extract_surface_mesh(mask_binary, step_size=2, mc_threshold=0.5):
    """
    Extracts a surface mesh from a binary mask using marching cubes.

    Parameters:
    - mask_binary: 3D numpy array (binary mask)
    - step_size: Step size for marching cubes
    - mc_threshold: Threshold value for marching cubes

    Returns:
    - verts: Vertices of the mesh
    - faces: Faces of the mesh
    """
    verts, faces, normals, values = marching_cubes(mask_binary, level=mc_threshold, step_size=step_size, allow_degenerate=False)
    return verts, faces


def decimate_mesh(verts, faces, target_reduction=0.5):
    """
    Decimates the mesh to reduce the number of triangles.

    Parameters:
    - verts: Vertices of the mesh
    - faces: Faces of the mesh
    - target_reduction: Fraction by which to reduce the mesh complexity

    Returns:
    - mesh_decimated: Decimated PyVista mesh
    """
    try:
        import pyvista as pv
    except ImportError:
        logging.error("PyVista is not installed. Please install it to use mesh decimation.")
        raise

    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int32)
    csf_mesh = pv.PolyData(verts, faces_pv)

    logging.info(f"Initial mesh has {csf_mesh.n_points} points and {csf_mesh.n_faces} faces.")
    print(f"Initial mesh has {csf_mesh.n_points} points and {csf_mesh.n_faces} faces.")
    print_memory_usage()
    log_memory_usage()

    csf_mesh_decimated = csf_mesh.decimate(target_reduction, inplace=False)

    logging.info(f"Decimated mesh has {csf_mesh_decimated.n_points} points and {csf_mesh_decimated.n_faces} faces.")
    print(f"Decimated mesh has {csf_mesh_decimated.n_points} points and {csf_mesh_decimated.n_faces} faces.")
    print_memory_usage()
    log_memory_usage()

    # Clean up
    del verts, faces, faces_pv, csf_mesh

    return csf_mesh_decimated


def visualize_and_save_html(original_img, denoised_img, refined_masks, k, output_html):
    """
    Visualizes the Original MRI, Denoised MRI, and CSF Masks using PyVista and saves as HTML.
    """
    print("\nStarting visualization with PyVista...")
    logging.info("Starting visualization with PyVista.")

    import pyvista as pv
    import matplotlib.colors as colors

    # Calculate the total number of subplots needed
    total_subplots = 2 + k  # 2 for MRI volumes, k for clusters

    # Determine the number of rows and columns
    import math
    cols = 3
    rows = math.ceil(total_subplots / cols)

    # Initialize PyVista Plotter
    plotter = pv.Plotter(shape=(rows, cols), title="MRI and CSF Masks Visualization", window_size=[1800, 300 * rows])

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
    plotter.add_text("Denoised MRI", position="upper_left", font_size=10)
    plotter.show_axes()

    # -----------------------------
    # 3+. CSF Mask Overlays
    # -----------------------------
    for cluster_idx in range(k):
        mesh = refined_masks.get(cluster_idx)
        if mesh is None:
            continue  # Skip if mask is not available

        print(f"Adding Mask {cluster_idx} mesh to visualization...")
        logging.info(f"Adding Mask {cluster_idx} mesh to visualization.")

        # Extract surface mesh
        mask_binary = (mesh > 0).astype(np.uint8)
        if np.max(mask_binary) == 0:
            logging.warning(f"Mask {cluster_idx} is empty. Skipping visualization.")
            print(f"Warning: Mask {cluster_idx} is empty. Skipping visualization.")
            continue
        
        verts, faces = extract_surface_mesh(mask_binary)

        # Decimate mesh
        mesh_decimated = decimate_mesh(verts, faces)

        # Determine subplot position
        subplot_idx = cluster_idx + 2  # Offset by 2 for the initial MRI volumes
        row = subplot_idx // cols
        col = subplot_idx % cols

        # Add mesh to plotter
        plotter.subplot(row, col)
        plotter.add_mesh(mesh_decimated, color="red", opacity=0.5, show_edges=False, label=f"Mask {cluster_idx}")
        plotter.add_text(f"Mask {cluster_idx}", position="upper_left", font_size=10)
        plotter.show_axes()

        # Clean up
        del mask_binary, mesh_decimated

    # -----------------------------
    # Optional: Hide unused subplots
    # -----------------------------
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            if idx >= total_subplots:
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

    # Clean up
    del plotter, refined_masks, original_img, denoised_img


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

        # Clean up
        del mask_binary, verts, faces, mesh

    # Clean up
    del refined_masks, denoised_img_normalized
    return decimated_meshes, mesh_labels


def identify_good_cluster(refined_masks, denoised_img_before_norm, denoised_img_normalized, min_peak_height=0.5, histogram_output_path='density_histogram.png'):
    """
    Identifies the good cluster based on density concentration, sums pixel values on the mask from both
    denoised_before_norm and denoised_normalized images, and saves a 2D histogram of the sums over the Z-axis.

    Parameters:
    - refined_masks: Dictionary of refined mask arrays.
    - denoised_img_before_norm: 3D numpy array of the denoised MRI image before normalization.
    - denoised_img_normalized: 3D numpy array of the denoised MRI image after normalization.
    - min_peak_height: Minimum height of peaks to be considered in density profiles.
    - histogram_output_path: File path to save the 2D histogram image.

    Returns:
    - good_cluster_idx: Integer representing the index of the good cluster.
    """
    logging.info("Identifying the good cluster based on density concentration.")
    print("Identifying the good cluster based on density concentration.")

    cluster_scores = {}
    sum_values = {}  # To store sum of pixel values for each cluster

    for cluster_idx, mask in refined_masks.items():
        # Compute density profiles using the original denoised image before normalization
        density_x = (denoised_img_normalized * mask).sum(axis=(0, 2))  # Sum over y and z axes
        density_y = (denoised_img_normalized * mask).sum(axis=(0, 1))  # Sum over x and z axes
        density_x = density_x[75:175]
        density_y = density_y[100:200]
        # Find peaks in density profiles
        peaks_x, properties_x = find_peaks(density_x, height=min_peak_height)
        peaks_y, properties_y = find_peaks(density_y, height=min_peak_height)

        #Plot and save the density profiles
        plt.figure(figsize=(10, 8))
        plt.plot(density_x, label='Density Profile along X-axis')
        plt.plot(density_y, label='Density Profile along Y-axis')
        plt.plot(peaks_x, density_x[peaks_x], "x", label='Peaks along X-axis')
        plt.plot(peaks_y, density_y[peaks_y], "x", label='Peaks along Y-axis')
        plt.legend()
        plt.title(f'Density Profiles for Cluster {cluster_idx}')
        plt.xlabel('Position')
        plt.ylabel('Density')
        plt.savefig(f'density_profiles_cluster_{cluster_idx}.png', dpi=300)
        plt.close()
        
        
        # Compute scores based on the number and height of peaks
        score_x = properties_x['peak_heights'].sum() if len(peaks_x) > 0 else 0
        score_y = properties_y['peak_heights'].sum() if len(peaks_y) > 0 else 0

        # Total score for the cluster
        total_score = score_x + score_y
        #cluster_scores[cluster_idx] = total_score / (density_x.sum() + density_y.sum())
        cluster_scores[cluster_idx] = total_score 

        logging.info(f"Cluster {cluster_idx}: Score = {total_score} (X: {score_x}, Y: {score_y})")
        print(f"  Cluster {cluster_idx}: Score = {total_score} (X: {score_x}, Y: {score_y})")

        # Sum pixel values on the mask for both images
        sum_original = denoised_img_before_norm[mask].sum()
        sum_normalized = denoised_img_normalized[mask].sum()
        sum_values[cluster_idx] = {'original_sum': sum_original, 'normalized_sum': sum_normalized}

        logging.info(f"Cluster {cluster_idx}: Original Sum = {sum_original}, Normalized Sum = {sum_normalized}")
        print(f"  Cluster {cluster_idx}: Original Sum = {sum_original}, Normalized Sum = {sum_normalized}")

    if not cluster_scores:
        logging.error("No clusters available to identify the good cluster.")
        print("Error: No clusters available to identify the good cluster.")
        sys.exit(1)

    # Identify the cluster with the highest score
    good_cluster_idx = max(cluster_scores, key=cluster_scores.get)
    logging.info(f"Good cluster identified as Cluster {good_cluster_idx} with score {cluster_scores[good_cluster_idx]}.")
    print(f"Good cluster identified as Cluster {good_cluster_idx} with score {cluster_scores[good_cluster_idx]}.")

    # Generate 2D histogram (x,z) of the sum of values over the Z-axis for the good cluster
    good_mask = refined_masks[good_cluster_idx]
    sum_over_z = denoised_img_before_norm.sum(axis=0)  # Sum over z-axis
    # Apply the mask by multiplying with the sum over z-axis
    sum_over_z_masked = sum_over_z * good_mask.sum(axis=0)

    plt.figure(figsize=(10, 8))
    plt.imshow(sum_over_z_masked, cmap='hot', interpolation='nearest', norm=colors.LogNorm())
    plt.colorbar(label='Sum of Pixel Values over Z-axis')
    plt.title(f'2D Histogram (x,z) of Sum of Values over Z-axis for Cluster {good_cluster_idx}')
    plt.xlabel('X-axis')
    plt.ylabel('Z-axis')
    plt.savefig(histogram_output_path, dpi=300)
    plt.close()

    print(f"2D histogram saved as '{histogram_output_path}'.")
    logging.info(f"2D histogram saved as '{histogram_output_path}'.")

    return good_cluster_idx


def remove_eyes_from_csf_mask(mask, eye_centers, eye_radius):
    """
    Removes two spherical regions (eyes) from the CSF mask.

    Parameters:
    - mask: 3D numpy array representing the binary mask.
    - eye_centers: List of tuples, each representing the (z, y, x) coordinates of an eye center.
    - eye_radius: Radius of the spherical region to remove.

    Returns:
    - refined_mask: 3D numpy array with eyes removed.
    """
    zz, yy, xx = np.ogrid[:mask.shape[0], :mask.shape[1], :mask.shape[2]]
    for center in eye_centers:
        zc, yc, xc = center
        distance = (zz - zc)**2 + (yy - yc)**2 + (xx - xc)**2
        mask[distance <= eye_radius**2] = 0
    return mask


def calculate_csf_volume(csf_mask, mri_image, voxel_dimensions):
    """
    Calculates the CSF volume in mm³ by applying the CSF mask on the original MRI image before standardization.
    Each voxel's contribution to the volume is weighted based on its intensity.

    Parameters:
    - csf_mask: 3D numpy array representing the binary CSF mask.
    - mri_image: 3D numpy array representing the original MRI image before standardization.
    - voxel_dimensions: Tuple of three floats indicating the physical size of each voxel in mm (z, y, x).

    Returns:
    - csf_volume_mm3: Float representing the total CSF volume in mm³.
    """
    try:
        # Ensure that the mask and MRI image have the same shape
        if csf_mask is None:
            # If the CSF mask is not provided, return full volume 
            csf_volume_mm3 = np.sum(mri_image > 0) * np.prod(voxel_dimensions)
            logging.info(f"Calculated CSF Volume: {csf_volume_mm3} mm³")
            print(f"Calculated CSF Volume: {csf_volume_mm3} mm³")
            return csf_volume_mm3
        
        if csf_mask.shape != mri_image.shape:
            raise ValueError("CSF mask and MRI image must have the same dimensions.")
        
        # Apply the CSF mask to the original MRI image
        masked_mri = mri_image * csf_mask  # Retain original intensities within the mask
        
        # Normalize intensities to [0, 1] based on the provided criteria
        # Assuming MRI intensities range from 0 to 255
        coefficients = masked_mri / 255.0
        coefficients = np.clip(coefficients, 0, 1)  # Ensure coefficients are within [0, 1]
        
        # Compute the volume of a single voxel in mm³
        voxel_volume = np.prod(voxel_dimensions)  # z * y * x
        
        # Calculate the total CSF volume by summing weighted voxel volumes
        csf_volume_mm3 = np.sum(coefficients) * voxel_volume
        
        logging.info(f"Calculated CSF Volume: {csf_volume_mm3} mm³")
        print(f"Calculated CSF Volume: {csf_volume_mm3} mm³")
        
        return csf_volume_mm3
    
    except Exception as e:
        logging.error(f"Error calculating CSF volume: {e}")
        print(f"Error calculating CSF volume: {e}")
        return 0.0


def save_good_cluster_mask(mask, output_path):
    """
    Saves the good cluster mask as a TIFF file.

    Parameters:
    - mask: 3D numpy array representing the binary mask.
    - output_path: String path where the TIFF file will be saved.
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Convert boolean mask to uint8 (0 and 255)
        mask_uint8 = (mask > 0).astype(np.uint8) * 255
        tiff.imwrite(output_path, mask_uint8, imagej=True)
        logging.info(f"Good cluster mask saved at '{output_path}'.")
        print(f"Good cluster mask saved at '{output_path}'.")
    except Exception as e:
        logging.error(f"Failed to save good cluster mask: {e}")
        print(f"Error: Failed to save good cluster mask: {e}")


def compute_density_profile(mask):
    """
    Computes the density profile of the mask along the x and y axes.

    Parameters:
    - mask: 3D numpy array representing the binary mask.

    Returns:
    - density_x: 1D numpy array representing density along the x-axis.
    - density_y: 1D numpy array representing density along the y-axis.
    """
    density_x = mask.sum(axis=(0, 2))  # Sum over y and z
    density_y = mask.sum(axis=(0, 1))  # Sum over x and z
    return density_x, density_y


def visualize_good_csf_mask_html(denoised_img, good_csf_mask, output_html, step_size=2, mc_threshold=0.5, decimate_reduction=0.5):
    """
    Visualizes the Good CSF Mask using PyVista and saves it as an HTML file.

    Parameters:
    - denoised_img: 3D numpy array of the denoised MRI image.
    - good_csf_mask: 3D numpy array of the refined good CSF mask.
    - output_html: Path to save the HTML visualization.
    - step_size: Step size for the marching cubes algorithm to reduce computation.
    - mc_threshold: Threshold value for marching cubes.
    - decimate_reduction: Fraction by which to reduce the mesh complexity.
    """
    print("\nStarting visualization of the Good CSF Mask with PyVista...")
    logging.info("Starting visualization of the Good CSF Mask with PyVista.")

    # Extract the surface mesh using marching cubes
    print("Extracting surface mesh from the Good CSF Mask...")
    verts, faces, normals, values = marching_cubes(good_csf_mask, level=mc_threshold, step_size=step_size, allow_degenerate=False)
    faces = faces.astype(np.int32)

    # Create a PyVista mesh
    mesh = pv.PolyData(verts, np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int32))

    # Decimate the mesh to reduce complexity
    print("Decimating the mesh to reduce complexity...")
    mesh_decimated = mesh.decimate(decimate_reduction, inplace=False)

    # Initialize the PyVista Plotter
    plotter = pv.Plotter(off_screen=True)  # off_screen=True for environments without display

    # Add the CSF mask mesh
    print("Adding the Good CSF Mask mesh to the visualization...")
    plotter.add_mesh(mesh_decimated, color="red", opacity=1, show_edges=False, label="Good CSF Mask")

    # Add a legend
    plotter.add_legend([["Good CSF Mask", "red"]], bcolor="white", border=True)

    # Set camera position for better visualization
    plotter.view_isometric()

    # Export the visualization to HTML
    print(f"Exporting the Good CSF Mask visualization to '{output_html}'...")
    logging.info(f"Exporting the Good CSF Mask visualization to '{output_html}'.")
    plotter.export_html(output_html)
    print(f"Visualization successfully saved to '{output_html}'.")
    logging.info(f"Visualization successfully saved to '{output_html}'.")

    # Close the plotter to free memory
    plotter.close()
    print("Visualization process completed.")
    logging.info("Visualization process completed.")

    # Clean up
    del verts, faces, mesh, mesh_decimated, plotter


def compute_surface_area(volume):
    """
    Computes the surface area of a 3D binary volume using marching cubes.

    Parameters:
    - volume: 3D numpy array (binary)

    Returns:
    - surface_area: Float representing the surface area
    """
    try:
        verts, faces, normals, values = marching_cubes(volume, level=0.5)
        # Calculate surface area using the mesh vertices and faces
        surface_area = 0.0
        for face in faces:
            # Get the vertices of the face
            v0, v1, v2 = verts[face]
            # Compute the area of the triangle
            edge1 = v1 - v0
            edge2 = v2 - v0
            cross_prod = np.cross(edge1, edge2)
            area = np.linalg.norm(cross_prod) / 2.0
            surface_area += area
        return surface_area
    except Exception as e:
        logging.error(f"Error computing surface area: {e}")
        return 0.0

def compute_surface_area(volume):
    """
    Computes the surface area of a 3D binary volume using marching cubes.

    Parameters:
    - volume: 3D numpy array (binary)

    Returns:
    - surface_area: Float representing the surface area
    """
    try:
        verts, faces, normals, values = marching_cubes(volume, level=0.5)
        # Calculate surface area using the mesh vertices and faces
        surface_area = 0.0
        for face in faces:
            # Get the vertices of the face
            v0, v1, v2 = verts[face]
            # Compute the area of the triangle
            edge1 = v1 - v0
            edge2 = v2 - v0
            cross_prod = np.cross(edge1, edge2)
            area = np.linalg.norm(cross_prod) / 2.0
            surface_area += area
        return surface_area
    except Exception as e:
        logging.error(f"Error computing surface area: {e}")
        return 0.0

def create_visualization(refined_csf_mask, z_range, y_range, x_range, visualization_path):
    """
    Creates and saves a 2D projection image with threshold lines for X and Y axes.

    Parameters:
    - refined_csf_mask: 3D numpy array of the CSF mask with eyes removed.
    - z_range: Tuple indicating the Z-axis range (min, max) where eyes are expected.
    - y_range: Tuple indicating the Y-axis range (min, max) where eyes are expected.
    - x_range: Tuple indicating the X-axis range (min, max) where eyes are expected.
    - visualization_path: File path to save the visualization JPEG file.
    """
    try:
        # Create a max projection along Z-axis
        projection = np.max(refined_csf_mask, axis=0)
        
        # Plot the projection
        plt.figure(figsize=(8, 8))
        plt.imshow(projection, cmap='gray')
        plt.title('Refined CSF Mask with Eyes Removed')
        plt.axis('off')
        
        # Draw threshold lines
        x_min, x_max = x_range
        y_min, y_max = y_range
        
        # Vertical lines for X-axis thresholds
        plt.axvline(x=x_min, color='red', linestyle='--', linewidth=1, label='X Thresholds')
        plt.axvline(x=x_max, color='red', linestyle='--', linewidth=1)
        
        # Horizontal lines for Y-axis thresholds
        plt.axhline(y=y_min, color='blue', linestyle='--', linewidth=1, label='Y Thresholds')
        plt.axhline(y=y_max, color='blue', linestyle='--', linewidth=1)
        
        # Add legend for threshold lines
        plt.legend(loc='upper right')
        
        # Save the visualization image
        plt.tight_layout()
        plt.savefig(visualization_path, dpi=300)
        plt.close()
        
        logging.info(f"Saved visualization image with threshold lines at '{visualization_path}'.")
        print(f"Saved visualization image with threshold lines at '{visualization_path}'.")
    except Exception as e:
        logging.error(f"Failed to create and save visualization image: {e}")
        print(f"Error: Failed to create and save visualization image: {e}")

def detect_and_remove_eyes(
    csf_mask,
    min_eye_volume=500,
    max_eye_volume=5000,
    sphericity_threshold=0.2,
    z_range=(0, 180),    # Tuple indicating the Z-axis range [min, max]
    y_range=(70, 150),    # Tuple indicating the Y-axis range [min, max]
    x_range=(30, 100),    # Tuple indicating the X-axis range [min, max]
    save_eyes_mask=True,
    eyes_mask_path='detected_eyes_mask.tif'
):
    """
    Detects eyes within the CSF mask based on connected components analysis and removes them.
    Additionally, saves a visualization JPEG with threshold lines for X and Y axes.

    Parameters:
    - csf_mask: 3D numpy array representing the binary CSF mask.
    - min_eye_volume: Minimum volume (in voxels) to consider a component as an eye.
    - max_eye_volume: Maximum volume (in voxels) to consider a component as an eye.
    - sphericity_threshold: Minimum sphericity to consider a component as an eye.
    - z_range: Tuple indicating the Z-axis range (min, max) where eyes are expected.
    - y_range: Tuple indicating the Y-axis range (min, max) where eyes are expected.
    - x_range: Tuple indicating the X-axis range (min, max) where eyes are expected.
    - save_eyes_mask: Boolean indicating whether to save the eyes mask.
    - eyes_mask_path: File path to save the detected eyes mask as a TIFF file.

    Returns:
    - refined_csf_mask: 3D numpy array of the CSF mask with eyes removed.
    - eyes_centroids: List of tuples representing the (Z, Y, X) coordinates of detected eyes.
    - eyes_mask: 3D numpy array representing the binary mask of detected eyes.
    """
    logging.info("Starting eye detection using connected components analysis.")
    print("Starting eye detection using connected components analysis.")
    
    # Label connected components
    labeled_mask, num_features = label(csf_mask, return_num=True, connectivity=1)
    logging.info(f"Number of connected components found: {num_features}")
    print(f"Number of connected components found: {num_features}")
    
    # Extract properties of labeled regions
    regions = regionprops(labeled_mask)
    
    eyes_centroids = []
    eyes_labels = []
    
    # Iterate through regions to identify eyes within the specified Z, Y, and X-axis ranges
    for region in regions:
        volume = region.area  # Number of voxels
        
        # Create a binary mask for the region
        region_mask = labeled_mask == region.label
        
        # Compute surface area using marching cubes
        surface_area = compute_surface_area(region_mask)
        
        # Avoid division by zero
        if surface_area == 0:
            sphericity = 0.0
        else:
            sphericity = (np.pi**(1/3) * (6 * volume)**(2/3)) / surface_area
        
        # Debugging: Log region properties
        logging.debug(
            f"Region Label: {region.label}, Volume: {volume}, Surface Area: {surface_area:.2f}, Sphericity: {sphericity:.2f}"
        )
        
        # Apply criteria and Z-Y-X axis range filtering
        if (
            min_eye_volume <= volume <= max_eye_volume and
            sphericity >= sphericity_threshold
        ):
            centroid = region.centroid  # (Z, Y, X)
            z, y, x = centroid
            if (z_range[0] <= z <= z_range[1]) and (y_range[0] <= y <= y_range[1]) and (x_range[0] <= x <= x_range[1]):
                eyes_centroids.append(tuple(map(int, centroid)))
                eyes_labels.append(region.label)
                logging.info(
                    f"Detected potential eye at centroid: {centroid} with Volume: {volume} and Sphericity: {sphericity:.2f}"
                )
                print(
                    f"Detected potential eye at centroid: {centroid} with Volume: {volume} and Sphericity: {sphericity:.2f}"
                )
            else:
                logging.debug(
                    f"Region Label: {region.label} centroid Y={y} or X={x} outside the expected ranges Y={y_range}, X={x_range}. Skipping."
                )
    
    # Determine the number of detected eyes
    num_eyes = len(eyes_centroids)
    
    if num_eyes < 2:
        # Fewer than two eyes detected; skip removal
        logging.warning(
            f"Expected to detect 2 eyes within Z-axis range {z_range}, Y-axis range {y_range}, and X-axis range {x_range}, but found {num_eyes}. Skipping eye removal."
        )
        print(
            f"Warning: Expected to detect 2 eyes within Z-axis range {z_range}, Y-axis range {y_range}, and X-axis range {x_range}, but found {num_eyes}. Skipping eye removal."
        )
        # Set eyes_mask to all zeros and keep csf_mask as is
        eyes_mask = np.zeros_like(csf_mask, dtype=np.uint8)
        refined_csf_mask = csf_mask.copy()
    
    elif num_eyes == 2:
        # Exactly two eyes detected; proceed with removal
        eyes_mask = np.isin(labeled_mask, eyes_labels).astype(np.uint8)
        
        if save_eyes_mask:
            try:
                # Save eyes mask as TIFF
                tiff.imwrite(eyes_mask_path, eyes_mask * 255, imagej=True)
                logging.info(f"Saved detected eyes mask at '{eyes_mask_path}'.")
                print(f"Saved detected eyes mask at '{eyes_mask_path}'.")
                
                # Save coordinates as JSON
                coords_path = os.path.splitext(eyes_mask_path)[0] + '_coords.json'
                with open(coords_path, 'w') as f:
                    json.dump(eyes_centroids, f)
                logging.info(f"Saved detected eyes coordinates at '{coords_path}'.")
                print(f"Saved detected eyes coordinates at '{coords_path}'.")
                
                # Save visualization with threshold lines
                visualization_path = os.path.splitext(eyes_mask_path)[0] + '_visualization.jpg'
                create_visualization(csf_mask, z_range, y_range, x_range, visualization_path)
            except Exception as e:
                logging.error(f"Failed to save eyes mask, coordinates, or visualization at '{eyes_mask_path}': {e}")
                print(f"Error: Failed to save eyes mask, coordinates, or visualization at '{eyes_mask_path}': {e}")
        
        # Remove eyes from CSF mask
        refined_csf_mask = csf_mask.copy()
        refined_csf_mask[eyes_mask == 1] = 0
        logging.info("Eyes have been removed from the CSF mask.")
        print("Eyes have been removed from the CSF mask.")
    
    else:
        # More than two eyes detected; select the best two based on spatial closeness within Z-Y-X ranges
        logging.info(
            f"Detected {num_eyes} potential eyes within Z-axis range {z_range}, Y-axis range {y_range}, and X-axis range {x_range}. Selecting the best two based on spatial closeness."
        )
        print(
            f"Detected {num_eyes} potential eyes within Z-axis range {z_range}, Y-axis range {y_range}, and X-axis range {x_range}. Selecting the best two based on spatial closeness."
        )
        
        # Define a threshold for Z-axis proximity (voxels)
        z_threshold = 2  # Adjust as needed
        
        # Initialize variables to store the best pair
        best_pair = None
        min_distance = np.inf
        
        # Iterate over all possible pairs to find the closest pair within Z-axis proximity
        for eye1, eye2 in itertools.combinations(eyes_centroids, 2):
            z1, y1, x1 = eye1
            z2, y2, x2 = eye2
            
            # Check if they are within the Z-axis proximity threshold
            if abs(z1 - z2) <= z_threshold:
                # Compute Euclidean distance in Y and X axes
                distance = np.sqrt((y1 - y2) ** 2 + (x1 - x2) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    best_pair = (eye1, eye2)
        
        if best_pair:
            # Assign the best pair
            selected_centroids = list(best_pair)
            selected_labels = [labeled_mask[tuple(centroid)] for centroid in selected_centroids]
            eyes_centroids = selected_centroids
            eyes_labels = selected_labels
            
            # Create eyes mask
            eyes_mask = np.isin(labeled_mask, eyes_labels).astype(np.uint8)
            
            if save_eyes_mask:
                try:
                    # Save eyes mask as TIFF
                    tiff.imwrite(eyes_mask_path, eyes_mask * 255, imagej=True)
                    logging.info(f"Saved detected eyes mask at '{eyes_mask_path}'.")
                    print(f"Saved detected eyes mask at '{eyes_mask_path}'.")
                    
                    # Save coordinates as JSON
                    coords_path = os.path.splitext(eyes_mask_path)[0] + '_coords.json'
                    with open(coords_path, 'w') as f:
                        json.dump(eyes_centroids, f)
                    logging.info(f"Saved detected eyes coordinates at '{coords_path}'.")
                    print(f"Saved detected eyes coordinates at '{coords_path}'.")
                    
                    # Save visualization with threshold lines
                    visualization_path = os.path.splitext(eyes_mask_path)[0] + '_visualization.jpg'
                    create_visualization(csf_mask, z_range, y_range, x_range, visualization_path)
                except Exception as e:
                    logging.error(f"Failed to save eyes mask, coordinates, or visualization at '{eyes_mask_path}': {e}")
                    print(f"Error: Failed to save eyes mask, coordinates, or visualization at '{eyes_mask_path}': {e}")
            
            # Remove eyes from CSF mask
            refined_csf_mask = csf_mask.copy()
            refined_csf_mask[eyes_mask == 1] = 0
            logging.info("Eyes have been removed from the CSF mask.")
            print("Eyes have been removed from the CSF mask.")
        
        else:
            # No suitable pair found based on Z-axis proximity; skip removal
            logging.warning(
                f"No suitable pair of eyes found based on Z-axis proximity within threshold {z_threshold}. Skipping eye removal."
            )
            print(
                f"Warning: No suitable pair of eyes found based on Z-axis proximity within threshold {z_threshold}. Skipping eye removal."
            )
            eyes_mask = np.zeros_like(csf_mask, dtype=np.uint8)
            refined_csf_mask = csf_mask.copy()
    
    return refined_csf_mask, eyes_centroids, eyes_mask



def remove_small_objects_refined(refined_csf_mask, min_size=5):
    """
    Removes small objects from the refined CSF mask.

    Parameters:
    - refined_csf_mask: 3D numpy array representing the refined CSF mask.
    - min_size: Minimum size (in voxels) of objects to keep.

    Returns:
    - refined_csf_mask_cleaned: 3D numpy array with small objects removed.
    """
    logging.info(f"Starting removal of small objects from the CSF mask with minimum size {min_size} voxels.")
    print(f"Removing small objects from the CSF mask with minimum size {min_size} voxels.")
    number_of_voxels = np.sum(refined_csf_mask)
    try:
        # Ensure the mask is boolean
        mask_bool = refined_csf_mask.astype(bool)
        
        # Remove small objects
        refined_csf_mask_cleaned = remove_small_objects(mask_bool, min_size=min_size, connectivity=1)
        
        # Convert back to the original data type (e.g., uint8)
        refined_csf_mask_cleaned = refined_csf_mask_cleaned.astype(refined_csf_mask.dtype)
        
        logging.info(f"Small objects removed from the CSF mask. New mask has {np.sum(refined_csf_mask_cleaned)} voxels.")
        print(f"Small objects removed from the CSF mask. New mask has {np.sum(refined_csf_mask_cleaned)} voxels.")
        print(f"Percentage of voxels removed: {((number_of_voxels - np.sum(refined_csf_mask_cleaned)) / number_of_voxels) * 100:.2f}%")
        print(f'Removed {number_of_voxels - np.sum(refined_csf_mask_cleaned)} voxels.')

    except Exception as e:
        logging.error(f"Error while removing small objects: {e}")
        print(f"Error: Failed to remove small objects from the CSF mask. Details: {e}")
        # Return the original mask if an error occurs
        refined_csf_mask_cleaned = refined_csf_mask.copy()
    
    return refined_csf_mask_cleaned

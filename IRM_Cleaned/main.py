import argparse
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
from skimage.morphology import ball

from utils import (
    calculate_csf_volume,
    compute_density_profile,
    compute_z_value,
    create_refined_masks,
    denoise_image,
    detect_and_remove_eyes,
    identify_good_cluster,
    load_image,
    log_memory_usage,
    normalize_intensity,
    perform_kmeans,
    plot_refined_masks_on_slices,
    save_good_cluster_mask,
    setup_logging,
    visualize_and_save_html,
    visualize_good_csf_mask_html,
    visualize_feature_distributions,
    analyze_feature_distributions,
    compute_neighborhood_statistics,
    print_memory_usage,
    remove_small_objects_refined
)

from config import (
    BATCH_SIZE,
    DENOISED_TIFF_PATH,
    K,
    MASKS_DIR,
    MIN_SIZE,
    ORIGINAL_MRI_PATH,
    OUTPUT_HTML,
    OUTPUT_SLICES_DIR,
    STRUCTURING_ELEMENT_RADIUS,
    PLOT_DIR
)


def main():
    # ----- Argument Parsing -----
    parser = argparse.ArgumentParser(description="MRI and CSF Mask Processing Pipeline")
    parser.add_argument('--html', action='store_true', help='Save visualization as HTML')
    parser.add_argument('--slices', type=int, default=0, help='Save n slices with matplotlib after K-Means')
    parser.add_argument('--log', action='store_true', help='Save logs to a .txt file')
    parser.add_argument('--FT_ANALYSIS', action='store_true', help='Perform feature distribution analysis')
    
    args = parser.parse_args()
    
    # ----- Setup Logging -----
    log_file = 'logs/execution.log'
    setup_logging(args.log, log_file)
    
    logging.info("Pipeline started.")
    print("Pipeline started.")
    
    # ----- Step 1: Load and Denoise the TIFF Image -----
    original_img = load_image(ORIGINAL_MRI_PATH)

    voxel_dimensions = (0.9765635, 0.9765635, 1.0)  # Example values; adjust as needed
    csf_volume_mm3 = calculate_csf_volume(
        csf_mask=None,
        mri_image=original_img,
        voxel_dimensions=voxel_dimensions
    )
    denoised_img = denoise_image(original_img, DENOISED_TIFF_PATH)
    
    # Free memory if original image is no longer needed
    del original_img
    # ----- Step 2: Intensity Normalization -----
    denoised_img_before_norm = denoised_img.copy()
    denoised_img_normalized = normalize_intensity(denoised_img)

    # denoised_img_normalized[denoised_img_normalized < 0] = 0
    del denoised_img  # Free memory

    # ----- Step 3: Use Only the Top Third of the MRI -----
    num_slices, height, width = denoised_img_normalized.shape
    top_third = height // 3
    denoised_img_normalized = denoised_img_normalized[:, :top_third, :]
    denoised_img_before_norm = denoised_img_before_norm[:, :top_third, :]
    logging.info(f"Using top third of the MRI data: new shape {denoised_img_normalized.shape}")
    print(f"Using top third of the MRI data: new shape {denoised_img_normalized.shape}")

    # ----- Step 4: Enhanced Feature Extraction -----
    neighborhood_size = 2  # 3x3x3 neighborhood
    neighborhood_mean, neighborhood_variance = compute_neighborhood_statistics(
        denoised_img_normalized,
        neighborhood_size=neighborhood_size
    )
    
    z_values = compute_z_value(denoised_img_normalized)

    # ----- Step 5: Prepare Features for K-Means -----
    print("Preparing features for K-Means clustering with spatial connectivity...")
    logging.info("Preparing features for K-Means clustering with spatial connectivity.")

    intensity_flat = denoised_img_normalized.flatten().reshape(-1, 1)
    mean_flat = neighborhood_mean.flatten().reshape(-1, 1)
    variance_flat = neighborhood_variance.flatten().reshape(-1, 1)
    z_flat = z_values.flatten().reshape(-1, 1)

    del neighborhood_mean, neighborhood_variance  # Free memory

    features = np.hstack((intensity_flat, mean_flat, variance_flat, z_flat)).astype(np.float32)
    logging.info(f"Features shape: {features.shape}")
    print(f"Features shape: {features.shape}")
    print_memory_usage()
    log_memory_usage()

    # ----- Step 6: Visualize Feature Distributions -----
    if args.FT_ANALYSIS:
        visualize_feature_distributions(features)

    # ----- Step 7: Analyze Feature Distributions (Optional) -----
    if args.FT_ANALYSIS:
        analyze_feature_distributions(features)

    # ----- Step 8: K-Means Clustering with Connectivity -----
    connectivity_weight = 0.2  # Adjustable parameter
    kmeans = perform_kmeans(
        features,
        denoised_img_shape=denoised_img_normalized.shape,
        k=K,
        batch_size=BATCH_SIZE,
        connectivity_weight=connectivity_weight
    )
    labels = kmeans.labels_
    clustered_img = labels.reshape(denoised_img_normalized.shape)
    logging.info("Cluster labels reshaped to image dimensions.")
    print("Clustering complete.")
    print_memory_usage()
    log_memory_usage()
    
    del features  # Free memory

    # ----- Step 9: Create Refined Masks -----
    selem = ball(STRUCTURING_ELEMENT_RADIUS)
    refined_masks, csf_cluster = create_refined_masks(
        clustered_img,
        denoised_img_normalized,
        k=K,
        selem=selem,
        min_size=MIN_SIZE,
        masks_dir=MASKS_DIR
    )
    del clustered_img  # Free memory

    # ----- Step 10: Visualization and Saving as HTML -----
    if args.html:
        visualize_and_save_html(
            denoised_img_normalized,
            denoised_img_normalized,
            refined_masks,
            K,
            OUTPUT_HTML
        )
    
    # ----- Step 11: Save Specified Slices -----
    if args.slices > 0:
        os.makedirs(OUTPUT_SLICES_DIR, exist_ok=True)
        
        def get_slice_indices(n, max_slice):
            """
            Returns a list of slice indices based on the requested number of slices.
            Evenly distributes slices within the available range.
            """
            return np.linspace(0, max_slice - 1, n, dtype=int).tolist()
        
        max_slice = denoised_img_normalized.shape[0]
        slice_indices = get_slice_indices(args.slices, max_slice)
        logging.info(f"Selected slice indices for saving: {slice_indices}")
        print(f"Selected slice indices for saving: {slice_indices}")
        
        plot_refined_masks_on_slices(
            refined_masks,
            denoised_img_normalized,
            slice_indices,
            K,
            OUTPUT_SLICES_DIR
        )
    
    # ----- Step 12: Cleanup -----
    print("Pipeline completed successfully.")
    logging.info("Pipeline completed successfully.")
    print_memory_usage()
    log_memory_usage()

    # ----- Step 13: Find the Good Cluster Using Density Profile -----
    logging.info("Selecting and refining the good mask based on density profile.")
    print("\n----- Step 13: Selecting and Refining the Good Mask -----")
    
    csf_mask = refined_masks.get(csf_cluster)
    if csf_mask is None:
        logging.error(f"CSF cluster {csf_cluster} not found in refined masks.")
        print(f"Error: CSF cluster {csf_cluster} not found in refined masks.")
        sys.exit(1)
    
    # density_x, density_y = compute_density_profile(csf_mask)
    # logging.info("Computed density profiles along x and y axes.")
    
    # good_cluster_idx = identify_good_cluster(
    #     refined_masks,
    #     denoised_img_before_norm,
    #     denoised_img_normalized,
    #     min_peak_height= 1000,
    #     histogram_output_path=os.path.join(PLOT_DIR, 'density_histogram.png')
    # )
    
    # good_cluster_mask = refined_masks.get(good_cluster_idx)
    # if good_cluster_mask is None:
    #     logging.error(f"Good cluster {good_cluster_idx} mask not found.")
    #     print(f"Error: Good cluster {good_cluster_idx} mask not found.")
    #     sys.exit(1)

    good_cluster_mask = csf_mask 
    good_cluster_idx = csf_cluster
    logging.info(f"Selected good cluster mask with index {good_cluster_idx}.")
    # ----- Step 14: Remove Eyes from the Good Cluster Mask -----
    logging.info("Removing eyes from the good cluster mask.")
    print("\n----- Step 14: Removing Eyes from the Good Cluster Mask -----")
    
    refined_csf_mask, eyes_centroids, eyes_mask = detect_and_remove_eyes(
        csf_mask=good_cluster_mask,
        min_eye_volume=1000,        # Adjust based on data
        max_eye_volume=8000,       # Adjust based on data
        sphericity_threshold=0.6,  # Adjust based on data
        save_eyes_mask=True,
        eyes_mask_path=os.path.join(MASKS_DIR, 'detected_eyes_mask.tif')
    )
    
    logging.info(f"Detected eyes at centroids: {eyes_centroids}")
    print(f"Detected eyes at centroids: {eyes_centroids}")

    # ----- Step 15: Visualization of Detected Eyes on 2D Projection -----
    projection = refined_csf_mask.max(axis=0)
    eyes_projection = eyes_mask.max(axis=0)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(projection, cmap='gray')
    plt.title('Refined CSF Mask with Eyes Removed')
    plt.axis('off')
    
    # Overlay detected eyes with masking to handle transparency
    plt.imshow(np.ma.masked_where(eyes_projection == 0, eyes_projection), cmap='cool', alpha=0.5)
    
    # Draw circles around detected eyes
    ax = plt.gca()
    for centroid in eyes_centroids:
        _, y, x = centroid  # Exclude Z for 2D overlay
        circle = plt.Circle((x, y), radius=10, edgecolor='cyan', facecolor='none', linewidth=2)
        ax.add_patch(circle)
    
    plt.tight_layout()
    overlaid_path = os.path.join(PLOT_DIR, 'refined_csf_with_eyes_removed.png')
    plt.savefig(overlaid_path, dpi=300)
    plt.close()
    
    logging.info(f"Saved overlaid refined CSF mask with eyes removed at '{overlaid_path}'.")
    print(f"Saved overlaid refined CSF mask with eyes removed at '{overlaid_path}'.")
    
    # Update the refined masks dictionary
    refined_masks[good_cluster_idx] = refined_csf_mask

    # Remove small elements 
    refined_csf_mask = remove_small_objects_refined(refined_csf_mask, min_size=200)
    # ----- Step 16: Save the Good Cluster Mask as TIFF -----
    logging.info("Saving the good cluster mask as TIFF.")
    print("\n----- Step 16: Saving the Good Cluster Mask -----")
    
    save_good_cluster_mask(
        refined_csf_mask,
        os.path.join(MASKS_DIR, "good_csf_mask.tif")
    )
    
    # ----- Step 17: Calculate the Volume of the CSF in the MRI -----
    logging.info("Calculating the volume of CSF in the MRI.")
    print("\n----- Step 17: Calculating CSF Volume -----")
    
    voxel_dimensions = (0.9765635, 0.9765635, 1.0)  # Example values; adjust as needed
    csf_volume_mm3 = calculate_csf_volume(
        csf_mask=refined_csf_mask,
        mri_image=denoised_img_before_norm,
        voxel_dimensions=voxel_dimensions
    )
    logging.info(f"Total CSF Volume: {csf_volume_mm3} mm³")
    print(f"Total CSF Volume: {csf_volume_mm3} mm³")
    
    # ----- Step 18: Final Visualization of the Good CSF Mask -----
    logging.info("Final visualization of the Good CSF Mask.")
    print("\n----- Step 18: Final Visualization of the Good CSF Mask -----")
    
    if args.html:
        visualize_good_csf_mask_html(
            denoised_img_before_norm,
            refined_csf_mask,
            "good_cluster_malade.html",
            step_size=2,
            mc_threshold=0.5,
            decimate_reduction=0.5
        )
    
    if args.slices > 0:
        plot_refined_masks_on_slices(
            {good_cluster_idx: refined_csf_mask},  # Only the refined Good CSF mask
            denoised_img_normalized,
            slice_indices,
            1,  # K=1 since only one mask is being visualized
            OUTPUT_SLICES_DIR
        )
    
    # ----- Final Cleanup -----
    logging.info("Final cleanup completed.")
    print("Final cleanup completed.")
    print_memory_usage()
    log_memory_usage()


if __name__ == "__main__":
    main()

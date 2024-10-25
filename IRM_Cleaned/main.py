import argparse
import os
import sys
import logging 
import numpy as np
from skimage.morphology import ball

from utils import (
    setup_logging,
    print_memory_usage,
    log_memory_usage,
    load_image,
    denoise_image,
    normalize_intensity,
    compute_neighborhood_statistics,
    visualize_feature_distributions,
    analyze_feature_distributions,
    perform_kmeans,
    create_refined_masks,
    plot_refined_masks_on_slices,
    visualize_and_save_html,
    extract_and_decimate_meshes
)
from config import (
    DENOISED_TIFF_PATH,
    ORIGINAL_MRI_PATH,
    MASKS_DIR,
    OUTPUT_HTML,
    OUTPUT_SLICES_DIR,
    K,
    BATCH_SIZE,
    STRUCTURING_ELEMENT_RADIUS,
    MIN_SIZE
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
    denoised_img = denoise_image(original_img, DENOISED_TIFF_PATH)
    
    # Optionally delete original_img to free memory if not needed
    del original_img

    # ----- Step 2: Intensity Normalization -----
    denoised_img_normalized = normalize_intensity(denoised_img)
    # Free memory
    del denoised_img

    # ----- Modify to use only the top half of the MRI -----
    # Assuming the MRI data has shape (num_slices, height, width)
    # We'll take the top half along the height dimension
    num_slices, height, width = denoised_img_normalized.shape
    denoised_img_normalized = denoised_img_normalized[:, :height//2, :]
    logging.info(f"Using top half of the MRI data: new shape {denoised_img_normalized.shape}")
    print(f"Using top half of the MRI data: new shape {denoised_img_normalized.shape}")

    # ----- Step 3: Enhanced Feature Extraction -----
    neighborhood_size = 3  # 3x3x3 neighborhood
    neighborhood_mean, neighborhood_variance = compute_neighborhood_statistics(
        denoised_img_normalized,
        neighborhood_size=neighborhood_size
    )
    
    # ----- Step 4: Prepare Features for K-Means -----
    print("Preparing features for K-Means clustering with spatial connectivity...")
    logging.info("Preparing features for K-Means clustering with spatial connectivity.")

    intensity_flat = denoised_img_normalized.flatten().reshape(-1, 1)
    mean_flat = neighborhood_mean.flatten().reshape(-1, 1)
    variance_flat = neighborhood_variance.flatten().reshape(-1, 1)

    # Clean up memory
    del neighborhood_mean
    del neighborhood_variance

    features = np.hstack((intensity_flat, mean_flat, variance_flat)).astype(np.float32)
    print(f"Features shape: {features.shape}")
    logging.info(f"Features shape: {features.shape}")
    print_memory_usage()
    log_memory_usage()

    # ----- Step 5: Visualize Feature Distributions -----
    visualize_feature_distributions(features)
    
    # ----- Step 5.1: Feature Distribution Analysis (Optional) -----
    if args.FT_ANALYSIS:
        analyze_feature_distributions(features)
    
    # ----- Step 6: K-Means Clustering with Connectivity -----
    connectivity_weight = 0.1  # You can adjust this value as needed
    kmeans = perform_kmeans(
        features,
        denoised_img_shape=denoised_img_normalized.shape,
        k=K,
        batch_size=BATCH_SIZE,
        connectivity_weight=connectivity_weight
    )
    labels = kmeans.labels_
    clustered_img = labels.reshape(denoised_img_normalized.shape)
    print("Cluster labels reshaped to image dimensions.")
    logging.info("Cluster labels reshaped to image dimensions.")
    print_memory_usage()
    log_memory_usage()
    
    # Clean up memory
    del features

    # ----- Step 7: Create Refined Masks -----
    selem = ball(STRUCTURING_ELEMENT_RADIUS)
    refined_masks, csf_cluster = create_refined_masks(
        clustered_img,
        denoised_img_normalized,
        k=K,
        selem=selem,
        min_size=MIN_SIZE,
        masks_dir=MASKS_DIR
    )
    # Clean up memory
    del clustered_img

    # ----- Step 8: Visualization and Saving as HTML -----
    if args.html:
        visualize_and_save_html(
            denoised_img_normalized,  # Since we deleted original_img
            denoised_img_normalized,
            refined_masks,
            K,
            OUTPUT_HTML
        )
    
    # ----- Step 9: Save Specified Slices -----
    if args.slices > 0:
        os.makedirs(OUTPUT_SLICES_DIR, exist_ok=True)
        
        # Define slice indices based on the number of slices requested
        def get_slice_indices(n, max_slice):
            """
            Returns a list of slice indices based on the requested number of slices.
            """
            # Evenly distribute slices within the available slices
            indices = np.linspace(0, max_slice - 1, n, dtype=int).tolist()
            return indices
        
        max_slice = denoised_img_normalized.shape[0]
        slice_indices = get_slice_indices(args.slices, max_slice)
        print(f"Selected slice indices for saving: {slice_indices}")
        logging.info(f"Selected slice indices for saving: {slice_indices}")
        
        # Plot and save the refined masks overlays for the selected slices
        plot_refined_masks_on_slices(
            refined_masks,
            denoised_img_normalized,
            slice_indices,
            K,
            OUTPUT_SLICES_DIR
        )
    
    # ----- Step 10: Cleanup -----
    print("Pipeline completed successfully.")
    logging.info("Pipeline completed successfully.")
    print_memory_usage()
    log_memory_usage()

if __name__ == "__main__":
    main()

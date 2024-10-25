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
    
    # ----- Step 2: Intensity Normalization -----
    denoised_img_normalized = normalize_intensity(denoised_img)
    
    # ----- Step 3: Enhanced Feature Extraction -----
    neighborhood_size = 3  # 3x3x3 neighborhood
    neighborhood_mean, neighborhood_variance = compute_neighborhood_statistics(
        denoised_img_normalized,
        neighborhood_size=neighborhood_size
    )
    
    # ----- Step 4: Prepare Features for K-Means -----
    print("Preparing features for K-Means clustering...")
    logging.info("Preparing features for K-Means clustering.")
    
    intensity_flat = denoised_img_normalized.flatten().reshape(-1, 1)
    mean_flat = neighborhood_mean.flatten().reshape(-1, 1)
    variance_flat = neighborhood_variance.flatten().reshape(-1, 1)
    
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
    
    # ----- Step 6: K-Means Clustering -----
    kmeans = perform_kmeans(features, k=K, batch_size=BATCH_SIZE)
    labels = kmeans.labels_
    clustered_img = labels.reshape(denoised_img_normalized.shape)
    print("Cluster labels reshaped to image dimensions.")
    logging.info("Cluster labels reshaped to image dimensions.")
    print_memory_usage()
    log_memory_usage()
    
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
    
    # ----- Step 8: Visualization and Saving as HTML -----
    if args.html:
        visualize_and_save_html(
            original_img,
            denoised_img_normalized,
            refined_masks,
            K,
            OUTPUT_HTML
        )
    
    # ----- Step 9: Save Specified Slices -----
    if args.slices > 0:
        os.makedirs(OUTPUT_SLICES_DIR, exist_ok=True)
        
        # Define slice indices based on the number of slices requested
        def get_slice_indices(n):
            """
            Returns a list of slice indices based on the requested number of slices.
            For N=1: [90]
            For N=2: [90, 120]
            For N=3: [60, 90, 120]
            """
            mapping = {
                1: [90],
                2: [90, 120],
                3: [60, 90, 120],
                4: [60, 90, 120, 150],
                5: [60, 90, 105, 120, 150],
            }
            return mapping.get(n, [90])  # Default to [90] if n not in mapping
        
        slice_indices = get_slice_indices(args.slices)
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

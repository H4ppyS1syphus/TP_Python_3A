# config.py

import os

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MASKS_DIR = os.path.join(BASE_DIR, 'masks')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
PLOT_DIR = os.path.join(BASE_DIR, 'plots')

# File Paths
ORIGINAL_MRI_PATH = os.path.join('mri_data', 'csf2.tif')  # Original MRI
DENOISED_MRI_PATH = os.path.join('mri_data', 'denoised_nl_2.tif')  # Denoised MRI
DENOISED_TIFF_PATH = os.path.join('mri_data', 'denoised_nl_2.tif')  # Path to save/load denoised TIFF

# Masks
MASK_FILENAMES = [
    'csf_mask_class_0_refined_2.tif',
    'csf_mask_class_1_refined_2.tif',
    'csf_mask_class_2_refined_2.tif',
    'csf_mask_class_3_refined_2.tif',
    'csf_mask_class_4_refined_2.tif'
]

# Visualization
OUTPUT_HTML = os.path.join(BASE_DIR, 'ALL_clusters.html')
OUTPUT_SLICES_DIR = os.path.join(BASE_DIR, 'slices')

# K-Means Parameters
K = 5  # Number of clusters
BATCH_SIZE = 10000  # For MiniBatchKMeans

# Morphological Operations
STRUCTURING_ELEMENT_RADIUS = 3  # Radius for structuring element
MIN_SIZE = 5  # Minimum size for small object removal

# Logging
DEFAULT_LOG_FILE = os.path.join(LOGS_DIR, 'execution.log')

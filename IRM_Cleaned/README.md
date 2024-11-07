# MRI and CSF Mask Processing Pipeline 🚀

## Table of Contents 📑

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Command-Line Arguments](#command-line-arguments)
  - [Examples](#examples)
- [Dependencies](#dependencies)
- [License](#license)

## Overview 🔍

The **MRI and CSF Mask Processing Pipeline** is a Python-based project for processing Magnetic Resonance Imaging (MRI) data and Cerebrospinal Fluid (CSF) masks. The pipeline performs the following tasks:

### **Key Functionalities:**

- 📥 **Image Loading and Denoising**: Loads original MRI images and applies Non-Local Means denoising to enhance image quality.
- 🎚️ **Intensity Normalization**: Standardizes intensity values across datasets for consistent analysis.
- 🧮 **Feature Extraction**: Computes neighborhood statistics (mean and variance) to enrich the feature set.
- 📊 **Clustering**: Segments MRI data into clusters using MiniBatch K-Means with spatial connectivity.
- 🧹 **Morphological Operations**: Cleans and refines segmented masks by removing noise and filling gaps.
- 🗺️ **Mesh Extraction and Decimation**: Generates surface meshes from binary masks and optimizes them for visualization.
- 🖥️ **Visualization**: Provides interactive 3D visualizations of MRI data and refined CSF masks using PyVista.
- 📈 **Feature Distribution Analysis (Optional)**: Utilizes seaborn pairplots for analyzing feature relationships.
- 💾 **Output Options**: Saves visualizations as HTML files, exports selected slices as images, and logs execution details.
- 🔄 **Memory Optimization**: Efficient memory management for handling large datasets.
- 📝 **Logging Support**: Detailed logs for monitoring and debugging pipeline execution.


This pipeline is modular and customizable, suitable for both research and clinical applications.

## Features 🌟

- **Non-Local Means Denoising**: Reduces noise while preserving structural details.
- **Standard Intensity Normalization**: Ensures consistent intensity across MRI slices.
- **Advanced Feature Extraction**: Captures contextual information using neighborhood statistics (mean, variance).
- **Efficient Clustering**: Scalable segmentation with MiniBatch K-Means.
- **Morphological Refinement**: Cleans segmented masks by removing artifacts and filling gaps.
- **3D Visualization**: Interactive 3D models for MRI and CSF data.
- **Feature Distribution Analysis**: Optional pairwise feature relationship plots for detailed analysis.
- **Flexible Output Options**: Supports saving visualizations, exporting slices, and logging.
- **Memory Optimization**: Efficient memory management for large datasets.
- **Logging Support**: Detailed logs for monitoring and debugging.

## Project Structure 📁
```bash
mri_project/
├── main.py                 # Entry point for the pipeline
├── utils.py                # Utility functions for processing tasks
├── config.py               # Configuration settings for file paths and parameters
├── requirements.txt        # List of dependencies
├── README.md               # Project documentation (this file)
├── logs/                   # Directory for log files
│   └── execution.log
├── masks/                  # Directory for refined CSF masks
├── slices/                 # Directory for saved slice images
├── playground.ipynb        # Jupyter notebook for interactive exploration
└── notebooks/              # Directory for additional notebooks
    └── playground.ipynb
```

main.py: Entry point of the project. Handles argument parsing and orchestrates the entire processing pipeline.
    utils.py: Contains utility functions for various processing tasks, including image loading, denoising, normalization, feature extraction, clustering, morphological operations, mesh processing, visualization, and logging.
    config.py: Stores configuration variables such as file paths, parameters, and constants used throughout the project.
    requirements.txt: Lists all Python dependencies required to run the project.
    README.md: Documentation of the project (this file).
    logs/: Directory to store log files when logging is enabled.
    masks/: Contains refined CSF mask TIFF files generated after clustering and morphological operations.
    slices/: Stores saved slice images with overlaid masks for detailed inspection.
    playground.ipynb: Jupyter Notebook providing an interactive environment to explore and experiment with the pipeline.
    notebooks/: Directory containing Jupyter Notebooks related to the project.

## Installation
 
 Clone the Repository

```bash
git clone https://github.com/yourusername/mri_project.git
cd mri_project
```

Create a Virtual Environment (Optional but Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install Dependencies

```bash

pip install -r requirements.txt
```

Prepare Directories

The script will automatically create necessary directories (logs/, masks/, slices/, notebooks/) if they do not exist. Ensure that your MRI and mask files are placed in the correct locations as specified in config.py.

## Usage

The pipeline is executed via the main.py script, which accepts various command-line arguments to customize its behavior.

### Command-Line Arguments

    --html: Saves the visualization as an interactive HTML file.
    --slices N: Saves N number of slices with matplotlib after K-Means clustering.
    --log: Enables logging of execution details and memory usage to a .txt file.
    --FT_ANALYSIS: Performs additional feature distribution analysis using seaborn's pairplot.

Examples

- Save Visualization as HTML Only

```bash
python main.py --html
```

Output:
Saves the interactive 3D visualization as mri_csf_visualization_2.html in the project directory.

Save 2 Slices with Matplotlib After K-Means

```bash
python main.py --slices 2
```

Output:
Saves overlaid images for slices 90 and 120 across all clusters in the slices/ directory. Total images saved: 2 slices × K clusters (e.g., 2 × 5 = 10 images).

Perform Feature Distribution Analysis

```
python main.py --FT_ANALYSIS
```

Output:
Generates and displays pairwise feature distribution plots using seaborn's pairplot. No files are saved.


## Dependencies

Ensure all dependencies are installed by running:

### Key Dependencies:

    Python Libraries:
        numpy: Numerical operations.
        matplotlib: Plotting and visualization.
        pyvista: 3D visualization.
        scikit-image: Image processing tasks.
        tifffile: Handling TIFF files.
        psutil: Monitoring system resources.
        tqdm: Progress bars.
        scikit-learn: Machine learning algorithms (e.g., K-Means).
        seaborn: Statistical data visualization.
        pandas: Data manipulation and analysis.
        ipywidgets: Interactive widgets for Jupyter Notebooks.


## License

This project is licensed under the MIT License.

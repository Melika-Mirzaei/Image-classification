# Superpixel and Keypoint Analysis

This repository contains Python code for analyzing images using superpixels and keypoints. The goal is to extract relevant features from images and create a graph-based representation.

## Strategies

1. **Superpixel Extraction (Strategy 1):**
   - Superpixels are obtained using the SLIC algorithm.
   - The input image is converted to RGB.
   - The number of superpixels is set to 100, and compactness is set to 10.
   - The resulting superpixels are stored in an array, along with the total count.

2. **Keypoint Extraction (Strategy 2):**
   - Keypoints are detected using the SIFT (Scale-Invariant Feature Transform) algorithm.
   - The number of keypoints is calculated.

## Coordinate Calculation

1. **Superpixel Coordinates (Strategy 1):**
   - For each superpixel, the centroid coordinates (x, y) are computed.
   - The mean intensity of each superpixel is also recorded.

2. **Keypoint Coordinates (Strategy 2):**
   - For each keypoint, the (x, y) coordinates are extracted.
   - The intensity value at each keypoint location is stored.

## Graph Creation

1. **Edge-Based Graph (Strategy 1):**
   - Nodes represent superpixel centroids.
   - Edges connect nodes if the Euclidean distance between centroids is below a specified threshold.

2. **Intensity-Based Graph (Strategy 2):**
   - Nodes represent keypoints.
   - Edges connect nodes if the absolute intensity difference between keypoints is below a threshold.

## Network Properties

- Degree centrality, closeness centrality, average neighbor degree, betweenness centrality, and PageRank are computed for the created graph.

## Usage

1. Install the required Python packages:
   ```
   pip install opencv-python numpy networkx scikit-image matplotlib pandas scikit-learn category-encoders
   ```

2. Run the provided code to extract superpixels, keypoints, and create graphs.

# Image Feature Extraction and Classification

This repository contains Python code for extracting features from grayscale images, creating graphs based on those features, and applying machine learning classifiers for image categorization. The project focuses on two main strategies: superpixel-based analysis and keypoint-based analysis.

## Strategies

### 1. Superpixel Extraction (Strategy 1)

- **Description**: Superpixels are extracted using the SLIC (Simple Linear Iterative Clustering) algorithm.
- **Steps**:
    - Convert the grayscale image to RGB format.
    - Use SLIC to segment the image into regions based on color similarity.
    - Calculate the number of superpixels.
- **Code Snippet**:
    ```python
    # Example usage
    segments, num_superpixels = extract_superpixels(image)
    ```

### 2. Keypoint Extraction (Strategy 2)

- **Description**: Keypoints are detected using OpenCV's SIFT (Scale-Invariant Feature Transform) algorithm.
- **Steps**:
    - Detect keypoints in the grayscale image.
    - Calculate the number of keypoints.
- **Code Snippet**:
    ```python
    # Example usage
    keypoints, num_keypoints = extract_keypoints(image)
    ```

## Graph Creation

- Two types of graphs are constructed based on extracted features:
    1. **Edge-Weighted Graphs** (Strategy 1 and 2):
        - Nodes represent superpixels or keypoints.
        - Edges are added if the distance between nodes is below a specified threshold.
    2. **Intensity-Weighted Graphs** (Strategy 1 and 2):
        - Nodes represent superpixels or keypoints.
        - Edges are added if the intensity difference between nodes is below a specified threshold.

## Network Properties

- Various network properties are extracted from the created graphs:
    - Degree centrality
    - Closeness centrality
    - Average neighbor degree
    - Betweenness centrality
    - PageRank

## Machine Learning Classification

- The extracted features are used for image classification.
- Various classifiers are evaluated:
    - Decision Tree
    - Random Forest
    - K-Nearest Neighbor
    - Bagging
    - Boosting
    - Naive Bayes
    - SVM
    - Logistic Regression
    - SGD
    - Voting Ensemble

## Usage

1. Ensure you have the necessary Python libraries installed (e.g., OpenCV, scikit-learn).
2. Organize your grayscale images in a folder.
3. Update the `folder_path` variable in the code snippet.
4. Run the script to extract features, create graphs, and evaluate classifiers.

Feel free to explore and adapt this code for your specific image analysis tasks!

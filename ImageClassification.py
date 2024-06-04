import os
import cv2
import numpy as np
import math
import csv
import networkx as nx
from skimage.measure import regionprops
from skimage.segmentation import slic
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from category_encoders import OrdinalEncoder
from sklearn.model_selection import train_test_split


#Strategy 1:
def extract_superpixels(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    segments = slic(image_rgb, n_segments=100, compactness=10)
    num_superpixels = len(np.unique(segments))
    return segments, num_superpixels

#Strategy 2:
def extract_keypoints(image):
    # Use OpenCV's SIFT algorithm to extract keypoints
    sift = cv2.SIFT_create()
    keypoints = sift.detect(image, None)
    num_keypoints = len(keypoints)
    return keypoints, num_keypoints

#Strategy 1:
def calculate_coordinatesS(segments, image):
    regions = regionprops(segments, intensity_image=image)
    nodesS = []
    intensitiesS = []
    for region in regions:
        y, x = region.centroid
        nodesS.append((x, y))
        intensitiesS.append(region.mean_intensity)
    return nodesS, intensitiesS

#Stategy 1:
# Extract coordinates of all keypoints
def calculate_coordinatesK(keypoints, image):
    nodesK = []
    intensitiesK = []

    for kp in keypoints:
        x, y = kp.pt
        nodesK.append((x, y))
        intensity = image[int(y), int(x)]
        intensitiesK.append(intensity)
    return nodesK, intensitiesK

def create_graphE(nodes, threshold, num):

    graph = nx.Graph()
    for i in range(num):
        graph.add_node(nodes[i])

    for i in range(num):
        for j in range(i + 1, num):
            node1 = nodes[i]
            node2 = nodes[j]
            x1, y1 = node1
            x2, y2 = node2
            distance = math.dist((x1, y1), (x2, y2))
            if distance < threshold:
                graph.add_edge(nodes[i], nodes[j])
    return graph

def create_graphI(nodes, threshold, intensities, num):

    graph = nx.Graph()
    for i in range(num):
        graph.add_node(nodes[i])
    for i in range(num):
        for j in range(i + 1, num):
             if abs(int(intensities[i]) - int(intensities[j])) < threshold:
                  graph.add_edge(nodes[i], nodes[j])
    return graph  
 
def extract_network_properties(graph):
    properties = []
    deg_centrality = list(nx.degree_centrality(graph).values())
    closeness_centrality = list(nx.closeness_centrality(graph).values())
    between_centrality = list(nx.betweenness_centrality(graph, normalized = True, endpoints = False).values())
    pr = list(nx.pagerank(graph, alpha = 0.8).values())
    average_neighbor_degree = list(nx.average_neighbor_degree(graph).values())

    properties=[deg_centrality , closeness_centrality ,average_neighbor_degree, between_centrality , pr]
    
    return properties



# Example usage
folder_path = r'path'
image_files = os.listdir(folder_path)

attribute_vector1, attribute_vector2=[],[]
attribute_vector3, attribute_vector4=[],[]

for file_name in image_files:
    image_path = os.path.join(folder_path, file_name)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    file_name_split = file_name.split(".")
    image_num = int(file_name_split[0])
    category = (image_num//100)+1
    T1 = 50
    T2 = 20
    segments, num_superpixels = extract_superpixels(image)
    nodesS, intensitiesS = calculate_coordinatesS(segments, image)
    keypoints, num_keypoints = extract_keypoints(image)
    nodesK, intensitiesK = calculate_coordinatesK(keypoints, image)

    graph1 = create_graphE(nodesS, T1, num_superpixels)
    graph2 = create_graphE(nodesK, T1, num_keypoints)

    graph3 = create_graphI(nodesS, T2, intensitiesS, num_superpixels)
    graph4 = create_graphI(nodesK, T2, intensitiesK, num_keypoints)


    properties1 = extract_network_properties(graph1)
    attribute_vector1.append( [category] + properties1)
    
    properties2 = extract_network_properties(graph2)
    attribute_vector2.append( [category] + properties2)

    properties3 = extract_network_properties(graph3)
    attribute_vector3.append( [category] + properties3)

    properties4 = extract_network_properties(graph4)
    attribute_vector4.append( [category] + properties4)


# CSV file
with open(r'path', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write the header line
            header = ['category'] + [f'Property_{i}' for i in range(1, 6)]
            writer.writerow(header)
            #attribute_vector = [file_name] + properties
            for i in range(len(attribute_vector1)):
                 writer.writerow(attribute_vector1[i]) 

with open(r'path', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write the header line
            header = ['category'] + [f'Property_{i}' for i in range(1, 6)]
            writer.writerow(header)
            #attribute_vector = [file_name] + properties
            for i in range(len(attribute_vector2)):
                 writer.writerow(attribute_vector2[i])               

with open(r'path', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write the header line
            header = ['category'] + [f'Property_{i}' for i in range(1, 6)]
            writer.writerow(header)
            #attribute_vector = [file_name] + properties
            for i in range(len(attribute_vector3)):
                 writer.writerow(attribute_vector3[i])

with open(r'path', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write the header line
            header = ['category'] + [f'Property_{i}' for i in range(1, 6)]
            writer.writerow(header)
            #attribute_vector = [file_name] + properties
            for i in range(len(attribute_vector4)):
                 writer.writerow(attribute_vector4[i])

def machine_learning():
    for i in range(1, 5):
         df = pd.read_csv(f'attribute_vectors{i}.csv')

         # Select features and target
         X = df[['Property_1', 'Property_2', 'Property_3', 'Property_4', 'Property_5']]
         y = df['category'] 

         # Encode categorical data 
         encoder = OrdinalEncoder()
         X = encoder.fit_transform(X)

         # Train test split
         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

         # Split the data into training and testing sets
         Classifiers=[
             DecisionTreeClassifier(),
             RandomForestClassifier(),
             KNeighborsClassifier(n_neighbors=7),
             BaggingClassifier(),
             AdaBoostClassifier(),
             GaussianNB(),
             SVC(),
             LogisticRegression(max_iter=5000),  # Increase max_iter to allow for more iterations
             SGDClassifier(),
             VotingClassifier(estimators=[('lr', LogisticRegression()), ('rf', RandomForestClassifier()), ('gnb', GaussianNB())])
         ]

         accuracylist=[]
         for i in range(len(Classifiers)):
             classifier = Classifiers[i]
             classifier.fit(X_train, y_train)

             # Make predictions on test set
             y_pred = classifier.predict(X_test)

             # Calculate accuracy 
             accuracy = accuracy_score(y_test, y_pred)
             accuracylist.append(accuracy)

         print(f"Decision Tree Accuracy: {accuracylist}")
         ac = pd.DataFrame({"Algorithm": ["Decision Tree", "Random Forest", "K-Nearest Neighbor", "Bagging", "Boosting", "Naive Bayes", "SVM", "Logistic Regression","SGD", "Voting"], "Accuracy": accuracylist})
         ac = ac.sort_values(by="Accuracy", ascending=False)
         print(ac)

machine_learning()
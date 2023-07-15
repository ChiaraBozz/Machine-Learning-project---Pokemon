import numpy as np
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv
from PIL import Image
import os

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from sklearn.svm import SVC
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import colorsys
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import random
from sklearn.multiclass import OneVsRestClassifier

# Set the random seed for PyTorch
torch.manual_seed(0)
# Set the random seed for Python's built-in random module
random.seed(0)
# Set the random seed for NumPy
np.random.seed(0)
from itertools import product
import math
from sklearn.pipeline import Pipeline
RAND_STATE = 0

def PCA_KNN(train_data, test_data, train_labels, test_labels, n_components, random_state, n_neighbors):
    pca = PCA(n_components=n_components, random_state=random_state)
    pca_train = pca.fit_transform(train_data)

    #pca.fit(test_data)
    pca_test = pca.transform(test_data)
    
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(pca_train, train_labels)
    y_pred = knn.predict(pca_test)

    accuracy = accuracy_score(test_labels, y_pred)
    print("Accuracy:", accuracy)

    cm = confusion_matrix(test_labels, y_pred)
    # Print and then perform and imshow of the confusion matrix.
    print(cm)
    plt.imshow(cm)

def KNN_hyperparameter_tuning(train_data, train_labels, test_data, test_labels, list_labels):
    n_neighbors = [3, 5, 7, 9, 11, round(math.sqrt(train_data.shape[0]))]
    weights = ['uniform', 'distance']
    metric = ['euclidean', 'manhattan', 'cosine', 'chebyshev', 'minkowski']

    results = []

    for n, w, m in product(n_neighbors, weights, metric):
        #print(str(n) + "-" + w + m)
        # Create the KNN classifier with the current hyperparameters
        knn = KNeighborsClassifier(n_neighbors=n, weights=w, metric=m)
        
        # Fit the model to the training data
        knn.fit(train_data, train_labels)
        
        # Make predictions on the test data
        predictions = knn.predict(test_data)
        
        # Calculate the accuracy
        accuracy = accuracy_score(test_labels, predictions)
        

        acc = round(accuracy_score(test_labels, predictions), 4)
        precision = round(precision_score(test_labels, predictions, average='macro', zero_division=1), 4)
        recall = round(recall_score(test_labels, predictions, average='macro'), 4)    
        f1 = f1_score(test_labels, predictions, average='macro')
        #print('Fully supervised results: Accuracy {}, Precision {}, Recall {}'.format(acc, precision, recall))

        results.append({'n_neighbors': n, 'weights': w, 'metric' : m, 'Accuracy': acc, 'Precision': precision, 'Recall': recall, 'F1-score': f1})
    
    knn_df = pd.DataFrame(results)
    knn_df.sort_values(by='Accuracy', ascending=False, inplace=True)

    print(knn_df)  

    best_row_index = knn_df['Accuracy'].idxmax()

    best_n_neighbors = knn_df.loc[best_row_index, 'n_neighbors']
    best_weights = knn_df.loc[best_row_index, 'weights']
    best_metric = knn_df.loc[best_row_index, 'metric']

    print(best_n_neighbors, best_weights, best_metric)

    knn = KNeighborsClassifier(n_neighbors=best_n_neighbors, weights=best_weights, metric=best_metric)
    knn.fit(train_data, train_labels)
    y_pred = knn.predict(test_data)

    accuracy = accuracy_score(test_labels, y_pred)
    print("Accuracy:", accuracy)
    precision = round(precision_score(test_labels, predictions, average='macro', zero_division=1), 4)
    recall = round(recall_score(test_labels, predictions, average='macro'), 4)
    f1 = f1_score(test_labels, predictions, average='macro')

    cm = confusion_matrix(test_labels, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list_labels, yticklabels=list_labels)

    # Customize plot
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    return {'n_neighbors': best_n_neighbors, 'weights': best_weights, 'metric' : best_metric, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-score': f1}

def SVM_hyperparameter_tuning(train_data, train_labels, test_data, test_labels, list_labels):
    #kernels = ["linear", "poly", "rbf", "sigmoid", "precomputed"]  Precomputed matrix must be a square matrix. Input is a 195x360 matrix.
    kernels = ["linear", "poly", "rbf", "sigmoid"]
    gammas = ["scale", "auto"]

    results = []

    for kernel in kernels:
        for gamma in gammas:
            #print(kernel + "-" + gamma)

            pseudo_classifier = OneVsRestClassifier(SVC(kernel=kernel, degree=1, C=1, gamma=gamma, probability=True)).fit(train_data, train_labels)
            
            predictions = pseudo_classifier.predict(test_data)

            #cm = confusion_matrix(test_labels, predictions)

            acc = round(accuracy_score(test_labels, predictions), 4)
            precision = round(precision_score(test_labels, predictions, average='macro', zero_division=1), 4)
            recall = round(recall_score(test_labels, predictions, average='macro'), 4)
            f1 = f1_score(test_labels, predictions, average='macro')
            #print('Fully supervised results: Accuracy {}, Precision {}, Recall {}'.format(acc, precision, recall))

            results.append({'Kernel': kernel, 'Gamma': gamma, 'Accuracy': acc, 'Precision': precision, 'Recall': recall, 'F1-score': f1})

    svm_df = pd.DataFrame(results)
    svm_df.sort_values(by='Accuracy', ascending=False, inplace=True)

    print(svm_df)   

    best_row_index = svm_df['Accuracy'].idxmax()

    best_kernel = svm_df.loc[best_row_index, 'Kernel']
    best_gamma = svm_df.loc[best_row_index, 'Gamma']

    print(best_kernel, best_gamma)

    pseudo_classifier = OneVsRestClassifier(SVC(kernel=best_kernel, degree=1, C=1, gamma=best_gamma, probability=True)).fit(train_data, train_labels)
    predictions = pseudo_classifier.predict(test_data)

    cm = confusion_matrix(test_labels, predictions)
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list_labels, yticklabels=list_labels)

    # Customize plot
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    acc = round(accuracy_score(test_labels, predictions), 4)
    precision = round(precision_score(test_labels, predictions, average='macro'), 4)
    recall = round(recall_score(test_labels, predictions, average='macro'), 4)
    print('Fully supervised results: Accuracy {}, Precision {}, Recall {}'.format(acc, precision, recall))
    f1 = f1_score(test_labels, predictions, average='macro')

    return {'Kernel': best_kernel, 'Gamma': best_gamma, 'Accuracy': acc, 'Precision': precision, 'Recall': recall, 'F1-score': f1}
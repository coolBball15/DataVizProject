import plotly.express as px
import numpy as np
from sklearn.decomposition import PCA

def pca_decomposition(activations):
    """Return the pca decomposition of the activations."""
    # Reshape the 4D array into a 2D matrix
    num_samples = activations.shape[0]
    num_features = np.prod(activations.shape[1:])
    data_2d = activations.reshape(num_samples, num_features)

    # Standardize the data (optional but recommended for PCA)
    mean = np.mean(data_2d, axis=0)
    std = np.std(data_2d, axis=0)
    standardized_data = (data_2d - mean) / std


    pca = PCA(n_components=2)
    pca_transform = pca.fit_transform(standardized_data)
    return pca_transform

def plot(pca_data, labels):
    """
    Plots the PCA data with labeled points.
    
    Parameters:
    - pca_data (numpy.ndarray): The PCA data to be plotted.
    - labels (numpy.ndarray): The labels for each data point.
    """

    fig = px.scatter(
        x=pca_data[:, 0],
        y=pca_data[:, 1], 
        color=labels 
        )
    return fig


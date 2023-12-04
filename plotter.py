import plotly.express as px
import numpy as np
from sklearn.decomposition import PCA

def pca_decomposition(activations):
    """Return the pca decomposition of the activations."""
    pca = PCA(n_components=2)
    pca_transform = pca.fit_transform(activations)
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


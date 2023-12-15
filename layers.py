import re
import keract
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

def get_all_layers(model) -> list:
    """
    Return a list of layer names.

    Parameters:
    model (object): The model object.

    Returns:
    list: A list of layer names.
    """
    return [layer.name for layer in model.layers]

def layer_types(model) -> list:
    """
    Return a list of layer types.

    Parameters:
    model (object): The model object.

    Returns:
    list: A list of layer types.
    """
    return [layer.__class__.__name__ for layer in model.layers]

import re

def extract_name_without_number(name):
    """
    Extracts the name from a string by removing any trailing numbers after an underscore.

    Args:
        name (str): The input string containing the name.

    Returns:
        str: The extracted name without any trailing numbers.

    Example:
        >>> extract_name_without_number("layer_1")
        'layer'
        >>> extract_name_without_number("background_2")
        'background'
    """
    pattern = re.compile(r'_(\d+)$')
    match = pattern.search(name)
    if match:
        return name[:match.start()]
    else:
        return name
    
def get_layer_names(model) -> list:
    """
    Return a list of layer names without the _number.

    Parameters:
    model (object): The model object containing the layers.

    Returns:
    list: A list of unique layer names without the _number.
    """
    all_layers = [extract_name_without_number(layer.name) for layer in model.layers]
    unique_layers = list(set(all_layers))
    return unique_layers

def get_layer_count(model, unique_name):
    """
    Return the count of layers with a specific unique name.

    Parameters:
    model (tf.keras.Model): The model to search for layers.
    unique_name (str): The unique name of the layers to count.

    Returns:
    int: The count of layers with the specified unique name.
    """
    original_names = [layer.name for layer in model.layers]
    processed_names = [extract_name_without_number(name) for name in original_names]
    return processed_names.count(unique_name)

def get_layer_activations(model, layer_name, data):
    """
    Return the activations of a specific layer.

    Parameters:
    model (tf.keras.Model): The trained model.
    layer_name (str): The name of the layer whose activations are to be retrieved.
    data (numpy.ndarray): The input data for which activations are to be computed.

    Returns:
    numpy.ndarray: The activations of the specified layer.
    """
    activations = keract.get_activations(model, data, layer_names=layer_name, nodes_to_evaluate=None, output_format='simple', nested=False, auto_compile=True)
    return activations[layer_name]
def get_layer_activations(model, layer_name, data):
    """Return the activations of a specific layer."""
    activations = keract.get_activations(model, data, layer_names = layer_name, nodes_to_evaluate=None, output_format='simple', nested=False, auto_compile=True)
    return activations[layer_name]
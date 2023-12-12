import re
import keract
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

def get_all_layers(model) -> list:
    """Return a list of layer names."""
    return [layer.name for layer in model.layers]

def layer_types(model) -> list:
    """Return a list of layer types."""
    return [layer.__class__.__name__ for layer in model.layers]

def extract_name_without_number(name):
    pattern = re.compile(r'_(\d+)$')
    match = pattern.search(name)
    if match:
        return name[:match.start()]
    else:
        return name
    
def get_layer_names(model) -> list:
    """Return a list of layer names without the _number."""
    all_layers = [extract_name_without_number(layer.name) for layer in model.layers]
    unique_layers = list(set(all_layers))
    return unique_layers

def get_layer_count(model, unique_name):
    """Return the count of layers with a specific unique name."""
    original_names = [layer.name for layer in model.layers]
    processed_names = [extract_name_without_number(name) for name in original_names]
    return processed_names.count(unique_name)

def get_layer_activations(model, layer_name, data):
    """Return the activations of a specific layer."""
    activations = keract.get_activations(model, data, layer_names = layer_name, nodes_to_evaluate=None, output_format='simple', nested=False, auto_compile=True)
    return activations[layer_name]
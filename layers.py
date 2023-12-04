import re
from cifar100vgg import cifar100vgg
import keract
from keras.datasets import cifar100
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

model = cifar100vgg(train=False)
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

#print(get_layer_count(model.model, 'conv2d'))
#print(get_all_layers(model.model))

def get_layer_activations(model, layer_name, data):
    """Return the activations of a specific layer."""
    activations = keract.get_activations(model, data, layer_names = layer_name, nodes_to_evaluate=None, output_format='simple', nested=False, auto_compile=True)
    return activations[layer_name]

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
test_activations = get_layer_activations(model.model, 'dense_1', x_test[0:15])
print(f'Activation shape: {test_activations.shape}')

def pca_decomposition(activations):
    """Return the pca decomposition of the activations."""
    pca = PCA(n_components=2)
    pca_transform = pca.fit_transform(activations)
    return pca_transform


pca_data = pca_decomposition(test_activations)
print(f'Labels of the first 15 test images: {y_test[0:15]}')
print(f'PCA shape: {pca_data.shape}')
#plt.plot(pca_data[:, 0], pca_data[:, 1], 'ro')
#plt.show()

# Do a pca on the activations of the last layer
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# import numpy as np
# from keras import backend as K

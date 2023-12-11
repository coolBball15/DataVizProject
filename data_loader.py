import pickle
from cifar100vgg import cifar100vgg
from cifar10vgg import cifar10vgg
from keras.datasets import cifar100, cifar10



def unpickle(file):
    '''
    Unpickle data from the given file.

    Parameters:
    - file (str): The file path.

    Returns:
    - dict: The unpickled data.
    '''
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

def pick_x(data, x):
    '''
    Pick x number of data points from the given data set.

    Parameters:
    - data (dict): The data set.
    - x (int): The number of data points to pick.

    Returns:
    - tuple: A tuple containing data, filenames, fine_labels, and coarse_labels.
    '''
    filenames = data[b'filenames'][:x]
    fine_labels = data[b'fine_labels'][:x]
    coarse_labels = data[b'coarse_labels'][:x]
    selected_data = data[b'data'][:x]
    return selected_data, filenames, fine_labels, coarse_labels


def load_label_names(model_name):
    '''
    Load the label names from the meta file.

    Returns:
    - tuple: A tuple containing fine_label_names and coarse_label_names.
    '''
    if model_name == 'cifar10':
        return load_cifar10_label_names()
    elif model_name == 'cifar100':
        return load_cifar100_label_names('data/meta')
    
def load_cifar100_label_names(meta_file):
    '''
    Load the label names from the meta file.

    Returns:
    - tuple: A tuple containing fine_label_names and coarse_label_names.
    '''
    with open(meta_file, "rb") as file:
        meta = pickle.load(file, encoding="bytes")
        fine_label_names = [label.decode("utf-8") for label in meta[b'fine_label_names']]
        coarse_label_names = [label.decode("utf-8") for label in meta[b'coarse_label_names']]
    return fine_label_names #, coarse_label_names

def load_cifar10_label_names():
    labelindex2word = {0:"airplane", 1:"automobile", 2: "bird", 3: "cat", 4: "deer", 
    5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"
}
    label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    return label_names

def load_model(model_name):
    '''
    Load the model from the given file.

    Parameters:
    - model_name (str): The model name.

    Returns:
    - keras.engine.sequential.Sequential: The model.
    '''
    if model_name == 'cifar10':
        return cifar10vgg(train=False)
    elif model_name == 'cifar100':
        return cifar100vgg(train=False)

def load_data(model_name):
    '''
    Load the data from the given file.

    Parameters:
    - model_name (str): The model name.

    Returns:
    - tuple: A tuple containing x_train, y_train, x_test, and y_test.
    '''
    if model_name == 'cifar10':
        return cifar10.load_data()
    elif model_name == 'cifar100':
        return cifar100.load_data()
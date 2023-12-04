import pickle

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

def load_label_names(meta_file):
    '''
    Load the label names from the meta file.

    Returns:
    - tuple: A tuple containing fine_label_names and coarse_label_names.
    '''
    with open(meta_file, "rb") as file:
        meta = pickle.load(file, encoding="bytes")
        fine_label_names = [label.decode("utf-8") for label in meta[b'fine_label_names']]
        coarse_label_names = [label.decode("utf-8") for label in meta[b'coarse_label_names']]
    return fine_label_names, coarse_label_names
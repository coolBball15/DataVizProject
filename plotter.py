
import plotly.express as px
import numpy as np
from sklearn.decomposition import PCA


def apply_pca(data, num_components=3):
    pca = PCA(n_components=num_components)
    return pca.fit_transform(data)

def map_colors(labels, label_mapping):
    colors = {}
    for i, label in enumerate(labels):
        if label not in colors.keys():
            colors[str(label)] = label_mapping(label)
    return colors

def get_pca_ranges(data_pca, factor=0.1):
    ranges = {
        'x_min': min(data_pca[:, 0]) * (1 - factor),
        'x_max': max(data_pca[:, 0]) * (1 + factor),
        'y_min': min(data_pca[:, 1]) * (1 - factor),
        'y_max': max(data_pca[:, 1]) * (1 + factor)
    }
    return ranges

def filter_data_by_labels(data, labels, selected_labels):
    '''
    Filter data and corresponding labels based on selected labels.

    Parameters:
    - data (numpy.ndarray): The input data array.
    - labels (list): List of labels corresponding to the data points.
    - selected_labels (list): List of selected labels to filter the data by.

    Returns:
    - tuple: A tuple containing:
        - numpy.ndarray: Filtered data array containing only data points with labels in selected_labels.
        - list: Filtered list of labels corresponding to the filtered data points.
    '''
    filtered_data = np.array([data[i] for i in range(len(data)) if labels[i] in selected_labels])
    filtered_labels = [labels[i] for i in range(len(labels)) if labels[i] in selected_labels]
    return filtered_data, filtered_labels

'''def plot_pca(data, labels, factor = 0.1, label_mapping, title, num_components=3):
    pca = PCA(n_components=num_components)
    data_pca = pca.fit_transform(data)
    colors = map_colors(labels, label_mapping)
    ranges = get_pca_ranges(data_pca, factor)
    fig = px.scatter(x=data_pca[:, 0], y=data_pca[:, 1], color=labels, color_discrete_map=colors)
    fig.update_layout(yaxis_range= [ranges['y_min'], ranges['y_max']])  # Adjust these values as needed
    fig.update_layout(xaxis_range=[ranges['x_min'], ranges['x_max']])
    fig.update_layout(title= title)
    return fig'''

#########################################################
def generate_colors(label):
    """
    Generate an HSL color representation based on the input label.

    Parameters:
    - label (int): The label for which to generate the color.

    Returns:
    - str: An HSL color representation in the format 'hsl(hue, saturation%, lightness%)'.
    """
    return f'hsl({int((label / 20) * 360)}, 100%, 50%)'

def plot_pca_figure(data_pca, labels, color_mapping, title, show_legend=True):
    """
    Plot a scatter plot of PCA-transformed data.

    Parameters:
    - data_pca (numpy.ndarray): PCA-transformed data.
    - labels (list): List of labels corresponding to the data points.
    - color_mapping (function): A function to map labels to colors.
    - title (str): Title for the plot.
    - show_legend (bool): Whether to display the legend.

    Returns:
    - plotly.graph_objects.Figure: The Plotly figure object.
    """
    str_labels = [str(item) for item in labels]
    if len(data_pca) == 0:
        fig = px.scatter()
    else:
        fig = px.scatter(x=data_pca[:, 0], y=data_pca[:, 1], color=str_labels, color_discrete_map=color_mapping)
    ranges = get_pca_ranges(data_pca)
    fig.update_layout(yaxis_range= [ranges['y_min'], ranges['y_max']])  # Adjust these values as needed
    fig.update_layout(xaxis_range=[ranges['x_min'], ranges['x_max']])
    fig.update_layout(title=title)
    fig.update_layout(showlegend=show_legend)
    return fig


def plot_super_figure(data, coarse_labels, checked_labels, label_dic, title='PCA of Image Data'):
    """
    Plot a super figure showing PCA-transformed data for selected coarse labels.

    Parameters:
    - data (numpy.ndarray): Input data.
    - coarse_labels (list): List of coarse labels corresponding to the data points.
    - checked_labels (list): List of selected coarse labels.
    - label_dic (dict): Dictionary mapping label names to indices.
    - title (str): Title for the plot.

    Returns:
    - plotly.graph_objects.Figure: The Plotly figure object.
    """
    selected_labels = [label_dic[coarse_label] for coarse_label in checked_labels]
    data_pca = apply_pca(data)
    filtered_data, filtered_labels = filter_data_by_labels(data_pca, coarse_labels, selected_labels)
    colors = generate_colors(filtered_labels)
    color_mapping = map_colors(filtered_labels, colors)
    return plot_pca_figure(filtered_data, filtered_labels, color_mapping, title)


def plot_sub_figure(data, coarse_labels, fine_labels, checked_label, label_dic, title='PCA of Image Data'):
    """
    Plot a sub figure showing PCA-transformed data for selected fine labels within a coarse label.

    Parameters:
    - data (numpy.ndarray): Input data.
    - coarse_labels (list): List of coarse labels corresponding to the data points.
    - fine_labels (list): List of fine labels corresponding to the data points.
    - checked_label (str): Selected coarse label.
    - label_dic (dict): Dictionary mapping label names to indices.
    - title (str): Title for the plot.

    Returns:
    - plotly.graph_objects.Figure: The Plotly figure object.
    """
    selected_labels = [fine_labels[i] for i in range(len(coarse_labels)) if coarse_labels[i] == label_dic[checked_label] and fine_labels[i] not in selected_labels]
    data_pca = apply_pca(data)
    filtered_data, filtered_labels = filter_data_by_labels(data_pca, fine_labels, selected_labels)
    colors = generate_colors(filtered_labels)
    color_mapping = map_colors(filtered_labels, colors)
    return plot_pca_figure(filtered_data, filtered_labels, color_mapping, title, show_legend=False)


#########################################################
'''def plot_super_figure(data, coarse_labels, needed_coarse):
    needed_labels = []
    for coarse in needed_coarse:
        needed_labels.append(dic[coarse])
    # Specify the number of components to keep (you can adjust this as needed)
    num_components = 3
    a = 0
    b = 1

    # Create a PCA object
    pca = PCA(n_components=num_components)

    # Fit and transform the data
    data_pca = pca.fit_transform(data)

    # Fetch our ranges
    # Fetch the x range
    x_min = min(data_pca[:, a]) * 0.9
    x_max = max(data_pca[:, a]) * 1.1

    # Fetch the y range
    y_min = min(data_pca[:, b]) * 0.9
    y_max = max(data_pca[:, b]) * 1.1

    # Now we only want to fetch the data with a label in the needed_coarse
    data_pca = np.array([data_pca[i] for i in range(len(data_pca)) if coarse_labels[i] in needed_labels])
    coarse_labels = [coarse_labels[i] for i in range(len(coarse_labels)) if coarse_labels[i] in needed_labels]

    # Define a custom color mapping function
    def map_colors_coarse(label):
        hue = int((label / 20) * 360)
        saturation = 100  # You can adjust the saturation as needed
        lightness = 50   # You can adjust the lightness as needed
        color = f'hsl({hue}, {saturation}%, {lightness}%)'  # Use HSL color format
        return color

    # Create a scatter plot with custom colors based on the tuple labels
    #colors = [map_colors_coarse([coarse_labels[i], fine_labels[i]]) for i in range(len(coarse_labels))]
    colors = {}
    for i in range(len(coarse_labels)):
        if (coarse_labels[i], fine_labels[i]) not in colors.keys():
            colors[str(coarse_labels[i])] = map_colors_coarse(coarse_labels[i])

    # Create a scatter plot of the PCA results
    str_coarse = [str(item) for item in coarse_labels]
    if len(data_pca) == 0:
        # No data to show 
        fig = px.scatter()
    else:
        fig = px.scatter(x = data_pca[:, a], y = data_pca[:, b], color=str_coarse, color_discrete_map=colors)

    fig.update_layout(yaxis_range=[y_min,y_max])
    fig.update_layout(xaxis_range=[x_min,x_max])
    fig.update_layout(title='PCA of Image Data')
    fig.update_layout(showlegend=False)
    return fig

def plot_sub_figure(data, for_coarse_label_name, coarse_labels, fine_labels):
    # Specify the number of components to keep (you can adjust this as needed)
    needed_labels = []
    for_coarse_label = dic[for_coarse_label_name]

    for i in range(len(coarse_labels)):
        coarse = coarse_labels[i]

        if coarse == for_coarse_label and fine_labels[i] not in  needed_labels:
            needed_labels.append(fine_labels[i])

    num_components = 3
    a = 0
    b = 1

    # Create a PCA object
    pca = PCA(n_components=num_components)

    # Fit and transform the data
    data_pca = pca.fit_transform(data)

    # Fetch our ranges
    # Fetch the x range
    x_min = min(data_pca[:, a]) * 0.9
    x_max = max(data_pca[:, a]) * 1.1

    # Fetch the y range
    y_min = min(data_pca[:, b]) * 0.9
    y_max = max(data_pca[:, b]) * 1.1

    # Now we only want to fetch the data with a label in the needed_coarse
    hover_data = [str(i) for i in range(len(data_pca)) if fine_labels[i] in needed_labels]
    data_pca = np.array([data_pca[i] for i in range(len(data_pca)) if fine_labels[i] in needed_labels])
    fine_labels = [fine_labels[i] for i in range(len(fine_labels)) if fine_labels[i] in needed_labels]
    

    # Define a custom color mapping function
    def map_colors_fine(label, needed_labels):
        label = needed_labels.index(label)
        hue = int((label / 5) * 360)
        saturation = 100  # You can adjust the saturation as needed
        lightness = 50   # You can adjust the lightness as needed
        color = f'hsl({hue}, {saturation}%, {lightness}%)'  # Use HSL color format
        return color

    colors = {}
    for i in range(len(fine_labels)):
        if (fine_labels[i], fine_labels[i]) not in colors.keys():
            colors[str(fine_labels[i])] = map_colors_fine(fine_labels[i], needed_labels)

    # Create a scatter plot of the PCA results
    str_coarse = [str(item) for item in fine_labels]
    for i,item in enumerate(str_coarse):
        str_coarse[i] = fine_label_names[int(item)]
    if len(data_pca) == 0:
        # No data to show 
        sub_fig = px.scatter()
    else:
        sub_fig = px.scatter(x = data_pca[:, a], 
                             y = data_pca[:, b], 
                             color=str_coarse, 
                             color_discrete_map=colors,
                             hover_name= hover_data)


    sub_fig.update_layout(yaxis_range=[y_min,y_max])
    sub_fig.update_layout(xaxis_range=[x_min,x_max])
    sub_fig.update_layout(title='PCA of Image Data')
    return sub_fig
'''
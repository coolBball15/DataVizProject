from dash import Dash, dcc, html
import plotly.express as px
import pickle
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def pick_x(x):
    filenames = data_dict[b'filenames'][:x]
    fine_labels = data_dict[b'fine_labels'][:x]
    coarse_labels = data_dict[b'coarse_labels'][:x]
    data = data_dict[b'data'][:x]
    return data, filenames, fine_labels, coarse_labels

def load_label_names():
    with open("data/meta", "rb") as file:
        meta = pickle.load(file, encoding="bytes")
        fine_label_names = [label.decode("utf-8") for label in meta[b'fine_label_names']]
    return fine_label_names

fine_label_names = load_label_names()
print("oi")
data_dict = unpickle("data/train")
data, filenames, fine_labels, coarse_labels = pick_x(600)
fine_label_names = load_label_names()
# Specify the number of components to keep (you can adjust this as needed)
num_components = 8

# Create a PCA object
pca = PCA(n_components=num_components)

# Fit and transform the data
data_pca = pca.fit_transform(data)

# Define a custom color mapping function
def map_colors(label):
    first_number = label[0]
    hue = (label[1] % 10) * 36  # Adjust the hue range as needed
    saturation = 50  # You can adjust the saturation as needed
    lightness = 50   # You can adjust the lightness as needed
    color = f'hsl({hue}, {saturation}%, {lightness}%)'  # Use HSL color format
    return color

# Create a scatter plot with custom colors based on the tuple labels
colors = [map_colors([coarse_labels[i], fine_labels[i]]) for i in range(len(coarse_labels))]
colors = {}
for i in range(len(coarse_labels)):
    if (coarse_labels[i], fine_labels[i]) not in colors.keys():
        colors[(coarse_labels[i], fine_labels[i])] = map_colors([coarse_labels[i], fine_labels[i]])

a = 0
b = 1
labels = [(coarse_labels[i], fine_label_names[fine_labels[i]]) for i in range(len(coarse_labels))]
# Create a scatter plot of the PCA results
fig = px.scatter(data_pca[:, a], data_pca[:, b], color=labels, color_discrete_map=colors)
fig.update_layout(title='PCA of Image Data')

app = Dash(__name__)

app.layout = html.Div([

    dcc.Markdown('''
    ## LaTeX in a Markdown component:

    This example uses the block delimiter:
    $$
    \\frac{1}{(\\sqrt{\\phi \\sqrt{5}}-\\phi) e^{\\frac25 \\pi}} =
    1+\\frac{e^{-2\\pi}} {1+\\frac{e^{-4\\pi}} {1+\\frac{e^{-6\\pi}}
    {1+\\frac{e^{-8\\pi}} {1+\\ldots} } } }
    $$

    This example uses the inline delimiter:
    $E^2=m^2c^4+p^2c^2$

    ## LaTeX in a Graph component:

    ''', mathjax=True),

    dcc.Graph(mathjax=True, figure=fig)]
)

if __name__ == '__main__':
    app.run(debug=True)

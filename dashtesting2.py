from dash import Dash, dcc, html
from dash.dependencies import Input, Output
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
        coarse_label_names = [label.decode("utf-8") for label in meta[b'coarse_label_names']]
    return fine_label_names, coarse_label_names

fine_label_names, coarse_label_names = load_label_names()
data_dict = unpickle("data/train")
data, filenames, fine_labels, coarse_labels = pick_x(600)

dic = {}
for i in range(len(coarse_label_names)):
    dic[coarse_label_names[i]] = i

def plot_super_figure(data, coarse_labels, needed_coarse):
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

fig = plot_super_figure(data, coarse_labels, coarse_label_names)
app = Dash(__name__)

@app.callback(
    Output('super_figure', 'figure'),
    [Input('checklist', 'value')]
)
def update_checklist_output(checked_values):
    # Each checked value is now in the checked_values variable, we only want to plot data with superclas in these check_values
    return plot_super_figure(data, coarse_labels, checked_values)


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

    dcc.Checklist(
        id = 'checklist',
        options = coarse_label_names,
        value = coarse_label_names),

    dcc.Graph(mathjax=True, figure=fig, id='super_figure')]
)

if __name__ == '__main__':
    app.run(debug=True)
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pickle
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import io
import base64


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
data, filenames, fine_labels, coarse_labels = pick_x(6000)

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
    sub_fig.update_layout(height = 600,width = 600)
    return sub_fig

fig = plot_super_figure(data, coarse_labels, coarse_label_names)
sub_fig = plot_sub_figure(data, coarse_label_names[0], coarse_labels, fine_labels)


# Reshape the flat array into a 3D array (height, width, channels)
image_shape = (32, 32, 3)  # Adjust the dimensions based on your actual image size and channels
image_data = data[0].reshape((3,1024)).T.reshape(image_shape)

# Convert NumPy array to image format
img = Image.fromarray((image_data * 255).astype('uint8'))
img_byte_array = io.BytesIO()
img.save(img_byte_array, format='PNG')
img_base64 = base64.b64encode(img_byte_array.getvalue()).decode('utf-8')


app = Dash(__name__)

@app.callback(
    Output('super_figure', 'figure'),
    [Input('checklist', 'value')]
)
def update_checklist_output(checked_values):
    # Each checked value is now in the checked_values variable, we only want to plot data with superclas in these check_values
    return plot_super_figure(data, coarse_labels, checked_values)


@app.callback(
    Output('sub_figure', 'figure'),
    [Input('dropdown', 'value')]
)
def update_dropdown_output(selected_value):
    return plot_sub_figure(data, selected_value, coarse_labels, fine_labels)
    
fig.update_layout(clickmode='event+select')

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

    ## PCA of all data based on coarse labels:

    ''', mathjax=True),

    html.Div([
        dcc.Checklist(
            id='checklist',
            options=coarse_label_names,
            value=coarse_label_names
        )
    ], style={'width': '30%', 'display': 'inline-block'}),

    html.Div([
        dcc.Graph(mathjax=True, figure=fig, id='super_figure')
    ], style={'width': '50%', 'display': 'inline-block','height': '100%'}),

    dcc.Markdown('''## PCA for all data from one coarse-label'''),

    dcc.Dropdown(
        id = 'dropdown',
        options= coarse_label_names,
        value= coarse_label_names[0]),

    html.Div([
        dcc.Graph(mathjax=True, figure=sub_fig, id='sub_figure')
    ],style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}),
    html.Img(
        id = 'selected_image',
        style={'width': '50%', 'display': 'block', 'margin': 'auto'},
        src = ''
    )
    ]
)

@app.callback(
    Output('selected_image', 'src'),
    Input('sub_figure', 'clickData'))
def display_selected_image(clickData):
    if clickData is None:
        # If no point is clicked, return a placeholder or default image
        return ''

    # Extract the image from the clicked point data
    selected_index = int(clickData['points'][0]['hovertext'])
    selected_image = data[selected_index].reshape((3,1024)).T.reshape(image_shape)

    # Convert NumPy array to image format
    selected_img = Image.fromarray((selected_image).astype('uint8'))
    selected_img_byte_array = io.BytesIO()
    selected_img.save(selected_img_byte_array, format='PNG')
    selected_img_base64 = base64.b64encode(selected_img_byte_array.getvalue()).decode('utf-8')

    # Return the base64-encoded image to update the 'src' attribute of the 'selected-image' component
    return f'data:image/png;base64,{selected_img_base64}'

if __name__ == '__main__':
    app.run(debug=True)

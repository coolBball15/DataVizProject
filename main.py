from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from plotter import pca_decomposition, plot, get_image
from layers import get_layer_activations, get_layer_names, get_layer_count, get_all_layers
from lime_plotter import get_image_LIME
from cifar100vgg import cifar100vgg
from keras.datasets import cifar100
import numpy as np
from data_loader import load_label_names

import plotly.express as px

data_points = 100

model = cifar100vgg(train=False)

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train = y_train.astype('int32')
y_test = y_test.astype('int32')

test_activations = get_layer_activations(model.model, 'activation_14', x_test[0:data_points])

fine_label_names, coarse_label_names = load_label_names('data/meta')

pca_data = pca_decomposition(test_activations)

current_fine_labels = [fine_label_names[label[0]] for label in y_test[0:data_points]]
fig = plot(pca_data, current_fine_labels)

# To do: Show layer names in dropdown menu
# To do: Show specific layer in second dropdown menu
# Show selected image on click, on the right side of the dahsboard

#print(f'Shape of data: {x_test[0].shape}')
#print(f'Label of data: {fine_label_names[y_test[0][0]]}')
image = get_image(x_test[2])

app = Dash(__name__)

#fig.update_layout(clickmode='event+select')

app.layout = html.Div([
    dcc.Markdown('''## PCA of all data based on coarse labels:'''),

    dcc.Dropdown(
        id='dropdown',
        options=get_all_layers(model.model),
        value=get_all_layers(model.model)[0]
    ),

    html.Div([
        # Graph Div
        html.Div([
            dcc.Graph(mathjax=True, figure=fig, id='plot')
        ], style={'width': '50%', 'display': 'inline-block', 'height': '100%'}),

        # Image Div
        html.Div([
            html.H1('Selected Image', style={'text-align': 'center', 'margin-top': '20px', 'margin-bottom': '10px'}),
            html.H2(id='image_title', style={'text-align': 'center', 'margin-top': '10px'}),
            html.Img(
                id='selected_image',
                style={'width': '50%', 'display': 'inline-block', 'margin': 'auto'},
                src=''
            ),
            html.Img(
                id='selected_image_LIME',
                style={'width': '50%', 'display': 'inline-block', 'margin': 'auto'},
                src=''
            ),
        ], style={'width': '50%', 'display': 'inline-block', 'height': '100%'}),
    ], style={'width': '100%', 'display': 'flex'}),

    html.Div(id='my-output'),
    html.Div(id='my-output2'),
])

@app.callback(
    Output('my-output', 'children'),
    Input('dropdown', 'value')
)
def print_layer(input_value):
    return 'Output: {}'.format(input_value)

@app.callback(
    Output('plot', 'figure'),
    Input('dropdown', 'value')
)
def plot_figure(input_value):
    """
    Plots a figure based on the input value.

    Parameters:
    input_value (int): The input value used to generate the figure.

    Returns:
    plotly.graph_objs._figure.Figure: The generated scatterplot figure.
    """
    test_activations = get_layer_activations(model.model, input_value, x_test[0:data_points])
    pca_data = pca_decomposition(test_activations)
    fig = plot(pca_data, current_fine_labels)
    return fig

@app.callback(
    Output('my-output2', 'children'),
    Input('dropdown', 'value')
)
def print_activatio_shape(input_value):
    test_activations = get_layer_activations(model.model, input_value, x_test[0:data_points])
    return 'Shape of activation: {}'.format(test_activations.shape)

@app.callback(
    [Output('selected_image', 'src'),
     Output('selected_image_LIME', 'src'),
     Output('image_title', 'children')],
    Input('plot', 'clickData'))
def display_selected_image(clickData):
    """
    Displays the selected image based on the click data.

    Parameters:
    - clickData (dict): The click data containing information about the clicked point.

    Returns:
    - selected_img (PIL.Image.Image): The selected image in PIL Image format.
    - label (str): The label of the selected image.
    """
    if clickData is None:
        # If no point is clicked, return a placeholder or default image
        return image, 'Image: '
    # Extract the image from the clicked point data
    selected_index = int(clickData['points'][0]['hovertext'])
    selected_image = x_test[selected_index]
    selected_label = y_test[selected_index]



    # Convert NumPy array to image format
    selected_img = get_image(selected_image)
    selected_img_LIME = get_image_LIME(selected_image)
    return selected_img, selected_img_LIME, f'Image: {fine_label_names[selected_label[0]]}'



'''@app.callback(
    Output('super_figure', 'figure'),
    [Input('checklist', 'value')]
)
def update_checklist_output(checked_values):
    # Each checked value is now in the checked_values variable, we only want to plot data with superclas in these check_values
    return plot_super_figure(data, coarse_labels, checked_values, coarse_name_dict)


@app.callback(
    Output('sub_figure', 'figure'),
    [Input('dropdown', 'value')]
)

def update_dropdown_output(selected_value):
    return plot_sub_figure(data, coarse_labels, fine_labels, selected_value, coarse_name_dict)
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
    return 
'''

if __name__ == '__main__':
    app.run(debug=True)
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
from plotter import pca_decomposition, plot, get_image, get_LIME_image
from layers import get_layer_activations, get_layer_names, get_layer_count, get_all_layers
import numpy as np
from data_loader import  load_model,load_label_names, load_data
from prediction import predict_image_class

import plotly.express as px

data_points = 100

current_model = 'cifar10'
model = load_model(current_model)
(x_train, y_train), (x_test, y_test) = load_data(model_name = current_model)

print(f'Prueba funcion image_class: {predict_image_class(model, x_test[0]).shape}')

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train = y_train.astype('int32')
y_test = y_test.astype('int32')

test_activations = get_layer_activations(model.model, 'dense_1', x_test[0:data_points])

#fine_label_names, coarse_label_names = load_label_names(current_model)

pca_data = pca_decomposition(test_activations)

label_names = load_label_names(current_model)

current_labels = [label_names[label[0]] for label in y_test[0:data_points]]

#current_fine_labels = [fine_label_names[label[0]] for label in y_test[0:data_points]]

fig = plot(pca_data, current_labels)

# To do: Show layer names in dropdown menu
# To do: Show specific layer in second dropdown menu
# Show selected image on click, on the right side of the dahsboard

#print(f'Shape of data: {x_test[0].shape}')
#print(f'Label of data: {fine_label_names[y_test[0][0]]}')
image = ''#get_image(x_test[2])

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

                # Image and Prediction Div
        html.Div([
            # Image Div
            html.Div([
                html.H1('Selected Image', style={'text-align': 'center', 'margin-top': '20px', 'margin-bottom': '10px'}),
                html.H2(id='image_title', style={'text-align': 'center', 'margin-top': '10px'}),
                html.Img(
                    id='selected_image',
                    style={'width': '30%', 'display': 'block', 'margin': 'auto'},
                    src=''
                ),
            ], style={'width': '100%', 'display': 'inline-block', 'text-align': 'center'}),

            # Prediction Probability Div
            html.Div(id='prediction_probability', style={'text-align': 'center', 'margin-top': '20px'}),
        ], style={'width': '50%', 'display': 'inline-block', 'height': '100%', 'text-align': 'center'}),
    ], style={'width': '100%', 'display': 'flex'}),
])
'''     html.Div([
        html.Button('Generate LIME explanation', id='generate_button', n_clicks=0),
        html.Img(
            id='lime_explanation',
            style={'width': '30%', 'display': 'block', 'margin': 'auto'},
            src=''
        ),
    #], style={'width': '100%', 'display': 'flex'}),
    html.Div(id='number_of_clicks',children=''),
    html.Div(id='my-output'),
    html.Div(id='my-output2'),'''



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
    fig = plot(pca_data, current_labels)
    return fig



@app.callback(
    [Output('selected_image', 'src'),
     Output('image_title', 'children')],
    Input('plot', 'clickData'),
    prevent_initial_call=True)
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
    return selected_img, f'Image: {label_names[selected_label[0]]}'

@app.callback(
    Output('prediction_probability', 'children'),
    Input('plot', 'clickData'),
    )
def display_prediction_probability(clickData):

    if clickData is None:
        # If no point is clicked, return a placeholder or default image
        return 'No image selected'
    selected_index = int(clickData['points'][0]['hovertext'])
    selected_image = x_test[selected_index]
    selected_label = y_test[selected_index]
    pred = predict_image_class(model, selected_image)
    label = label_names[pred[0]]
    return f'Predicted label: {label}'
'''
@app.callback(
    Output('my-output', 'children'),
    Input('dropdown', 'value')
)
def print_layer(input_value):
    return 'Output: {}'.format(input_value)

@app.callback(
    Output('my-output2', 'children'),
    Input('dropdown', 'value')
)
def print_activatio_shape(input_value):
    test_activations = get_layer_activations(model.model, input_value, x_test[0:data_points])
    return 'Shape of activation: {}'.format(test_activations.shape)


@app.callback(
    Output('lime_explanation','src'),
    Output('number_of_clicks', 'children'),
    Input('generate_button', 'n_clicks'),
    Input('plot', 'clickData'),
    )
def generate_lime_explanation(n_clicks, clickData):
    selected_index = int(clickData['points'][0]['hovertext'])
    selected_image = x_test[selected_index]
    selected_label = y_test[selected_index]
    print(f'Shape of image: {selected_image.shape}')
    print(f'Number of clicks: {n_clicks}')
    if clickData is None:
        return '', f'Number of clicks{n_clicks}'
    
    return get_image(get_LIME_image(selected_image, model)), f'Number of clicks{n_clicks}'''


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
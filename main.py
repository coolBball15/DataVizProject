from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
from plotter import pca_decomposition, plot, get_image, get_image_LIME
from layers import get_layer_activations, get_all_layers
from data_loader import  load_model,load_label_names, load_data
from predicter import gen_prod_figure, predict_image_class
import tensorflow as tf


data_points = 1200

current_model = 'cifar10'
model = load_model(current_model)   

all_models = {}
tf.keras.backend.clear_session()
all_models['cifar10'] = load_model('cifar10')
tf.keras.backend.clear_session()
all_models['cifar100'] = load_model('cifar100')

loaded_data = {
    'cifar10': load_data('cifar10'),
    'cifar100': load_data('cifar100')
}

(x_train, y_train), (x_test, y_test) = loaded_data[current_model]

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train = y_train.astype('int32')
y_test = y_test.astype('int32')

test_activations = get_layer_activations(model.model, 'activation_14', x_test[0:data_points])

# Dictionary to hold label names for each model
model_label_names = {
    'cifar10': load_label_names('cifar10'),
    'cifar100': load_label_names('cifar100'),
    # ... other models
}
y_test_cifar10 = loaded_data['cifar10'][1][1].astype('int32')
y_test_cifar100 = loaded_data['cifar100'][1][1].astype('int32')

current_labels = {
    'cifar10': [model_label_names['cifar10'][label[0]] for label in y_test_cifar10[0:data_points]],
    'cifar100': [model_label_names['cifar100'][label[0]] for label in y_test_cifar100[0:data_points]]
}
label_names = load_label_names(current_model)
current_labels_cifar10 = [label_names[label[0]] for label in y_test[0:data_points]]


#fine_label_names, coarse_label_names = load_label_names(current_model)

pca_data = pca_decomposition(test_activations)


#current_fine_labels = [fine_label_names[label[0]] for label in y_test[0:data_points]]

fig = plot(pca_data, current_labels[current_model])

# To do: Show layer names in dropdown menu
# To do: Show specific layer in second dropdown menu
# Show selected image on click, on the right side of the dahsboard

#print(f'Shape of data: {x_test[0].shape}')
#print(f'Label of data: {fine_label_names[y_test[0][0]]}')
image = ''#get_image(x_test[2])

app = Dash(__name__)

#fig.update_layout(clickmode='event+select')

app.layout = html.Div([
    dcc.Markdown('''## Model dashboard:'''),
    dcc.Markdown('''### Select model:'''),
    dcc.Dropdown(
        id='model_selection_dropdown',
        options=[
            {'label': 'CIFAR10', 'value': 'cifar10'},
            {'label': 'CIFAR100', 'value': 'cifar100'},
        ],
        value='cifar10'
    ),
    dcc.Markdown('''### Select layer:'''),
    dcc.Dropdown(
        id='dropdown',
        options=get_all_layers(model.model),
        value=get_all_layers(model.model)[-1]
    ),

    # Main Content Div
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
            html.Div([
                html.H2(id='prediction_probability', style={'text-align': 'center', 'margin-top': '20px'}),
                dcc.Graph(id='prob_figure'),
            ], style={'width': '100%', 'display': 'inline-block', 'text-align': 'center'}),
        ], style={'width': '50%', 'display': 'inline-block', 'height': '100%', 'text-align': 'center'}),
    ], style={'width': '100%', 'display': 'flex'}),

    # LIME Explanation Div
    html.Div([
        html.Button('LIME explanation', id='button', style={'text-align': 'center', 'margin-bottom': '10px'}),
        html.Img(
            id='selected_image_LIME',
            style={'width': '50%', 'display': 'block', 'margin': 'auto'},
            src=''
        ),
    ], style={'width': '100%', 'display': 'block', 'text-align': 'center', 'margin-top': '20px'}),
])




@app.callback(
    Output('plot', 'figure'),
    [Input('dropdown', 'value'),
    Input('model_selection_dropdown', 'value')]
)
def plot_figure(input_value, selected_model):
    """
    Plots a figure based on the input value.

    Parameters:
    input_value (int): The input value used to generate the figure.

    Returns:
    plotly.graph_objs._figure.Figure: The generated scatterplot figure.
    """
    model = all_models[selected_model] 
    (x_train, y_train), (x_test, y_test) = loaded_data[selected_model]
    test_activations = get_layer_activations(model.model, input_value, x_test[0:data_points])
    pca_data = pca_decomposition(test_activations)
    fig = plot(pca_data, current_labels[selected_model])
    return fig


@app.callback(
    [Output('selected_image', 'src'),
     Output('image_title', 'children')],
    [Input('plot', 'clickData'),
        Input('model_selection_dropdown', 'value')],
    prevent_initial_call=True)
def display_selected_image(clickData, selected_model):
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
    (x_train, y_train), (x_test, y_test) = loaded_data[selected_model]
    selected_index = int(clickData['points'][0]['hovertext'])
    selected_image = x_test[selected_index]
    selected_label = y_test[selected_index]
    # Convert NumPy array to image format
    selected_img = get_image(selected_image)
    return selected_img, f'Image: {model_label_names[selected_model][selected_label[0]]}'

@app.callback(
    [Output('prediction_probability', 'children'),
    Output('prob_figure', 'figure')],
    [Input('plot', 'clickData'),
    Input('model_selection_dropdown', 'value')],
    )
def display_prediction_probability(clickData, selected_model):
    """
    Displays the predicted label and probability figure for a selected image.

    Parameters:
    - clickData (dict): The click data containing information about the selected image.
    - selected_model (str): The name of the selected model.

    Returns:
    - str: The predicted label.
    - figure: The probability figure.
    """
    model = all_models[selected_model] 
    (x_train, y_train), (x_test, y_test) = loaded_data[selected_model]
    if clickData is None:
        return 'No image selected', gen_prod_figure(0,selected_model)
    selected_index = int(clickData['points'][0]['hovertext'])
    selected_image = x_test[selected_index]
    selected_label = y_test[selected_index]
    pred, pred_dict = predict_image_class(model, selected_image, selected_model)
    prob_figure = gen_prod_figure(pred_dict,selected_model)
    label = model_label_names[selected_model][pred[0]]
    return f'Predicted label: {label}',prob_figure
@app.callback(
    Output('selected_image_LIME', 'src'),
    Input('button', 'n_clicks'),
    State('plot', 'clickData'),
    prevent_initial_call=True)
def display_lime_image(n_clicks, clickData):
    """
    Displays the LIME image corresponding to the selected data point.

    Parameters:
    - n_clicks (int): The number of times the image has been clicked.
    - clickData (dict): The data associated with the click event.

    Returns:
    - selected_img_LIME: The LIME image corresponding to the selected data point.
    """
    selected_index = int(clickData['points'][0]['hovertext'])
    selected_image = x_test[selected_index]

    selected_img_LIME = get_image(get_image_LIME(model, selected_image))
    return selected_img_LIME

if __name__ == '__main__':
    app.run(debug=True)


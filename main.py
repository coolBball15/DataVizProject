from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from plotter import pca_decomposition, plot
from layers import get_layer_activations, get_layer_names, get_layer_count, get_all_layers
from cifar100vgg import cifar100vgg
from keras.datasets import cifar100
import numpy as np
from data_loader import load_label_names
import plotly.express as px

data_points = 15

model = cifar100vgg(train=False)

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train = y_train.astype('int32')
y_test = y_test.astype('int32')

test_activations = get_layer_activations(model.model, 'dense_1', x_test[0:data_points])
print(f'Activation shape: {test_activations.shape}')

fine_label_names, coarse_label_names = load_label_names('data/meta')

pca_data = pca_decomposition(test_activations)

current_fine_labels = [fine_label_names[label[0]] for label in y_test[0:data_points]]
print(f'Fine label names of the first 15 test images: {current_fine_labels}')
print(f'PCA shape: {pca_data.shape}')

fig = plot(pca_data, current_fine_labels)


# To do: Show layer names in dropdown menu
# To do: Show specific layer in second dropdown menu


app = Dash(__name__)

#fig.update_layout(clickmode='event+select')

app.layout = html.Div([

    dcc.Markdown('''## PCA of all data based on coarse labels:'''),

    dcc.Dropdown(
        id = 'dropdown',
        options= get_all_layers(model.model),
        value= get_all_layers(model.model)[0]
        ),

    html.Div([
        dcc.Graph(mathjax=True, figure=fig, id='plot')
    ], style={'width': '50%', 'display': 'inline-block','height': '100%'}),
        
    ]
)

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
    return f'data:image/png;base64,{selected_img_base64}'''

if __name__ == '__main__':
    app.run(debug=True)
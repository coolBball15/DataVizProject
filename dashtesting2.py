from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import io
import base64
from data_loader import load_label_names, unpickle, pick_x
from plotter import plot_super_figure, plot_sub_figure

data_file = "data/train"
meta_file = "data/meta"
num_data_points = 600

# Load data and labels
fine_label_names, coarse_label_names = load_label_names(meta_file)
data_dict = unpickle(data_file)
data, filenames, fine_labels, coarse_labels = pick_x(data_dict, num_data_points)

# Create a dictionary for coarse label names
coarse_name_dict = {coarse_label_names[i]: i for i in range(len(coarse_label_names))}

fig = plot_super_figure(data, coarse_labels, coarse_label_names, coarse_name_dict)
sub_fig = plot_sub_figure(data, coarse_labels,fine_labels, coarse_label_names[0], coarse_name_dict)


# Reshape the flat array into a 3D array (height, width, channels)
image_shape = (32, 32, 3)  # Adjust the dimensions based on your actual image size and channels
image_data = data[0].reshape((3,1024)).T.reshape(image_shape)

# Convert NumPy array to image format
img = Image.fromarray((image_data * 255).astype('uint8'))
img_byte_array = io.BytesIO()
img.save(img_byte_array, format='PNG')
img_base64 = base64.b64encode(img_byte_array.getvalue()).decode('utf-8')


app = Dash(__name__)

fig.update_layout(clickmode='event+select')

app.layout = html.Div([

    dcc.Markdown('''## PCA of all data based on coarse labels:'''),
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
    # Add the fine label name as a title to the image

    html.H1('Selected Image',style={'text-align': 'center', 'margin-top': '20px', 'margin-bottom': '10px'}),
    html.H2(id='image_title', style={'text-align': 'center', 'margin-top': '10px'}),
    html.Div([
        html.Img(
            id='selected_image',
            style={'width': '50%', 'display': 'block', 'margin': 'auto'},
            src=''
        )
        
    ])]
)

@app.callback(
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
    return f'data:image/png;base64,{selected_img_base64}'

if __name__ == '__main__':
    app.run(debug=True)

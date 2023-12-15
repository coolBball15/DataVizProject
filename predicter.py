import numpy as np
from keras.preprocessing.image import img_to_array
import plotly.graph_objects as go

from data_loader import load_label_names

def predict_image_class(model, image, current_model):
    """
    Predicts the class of an image using a given model.

    Parameters:
    - model: The trained model used for prediction.
    - image: The image to be classified.
    - current_model: The current model being used.

    Returns:
    - class_index: The index of the predicted class.
    - ret_dict: A dictionary containing the top 5 predicted labels and their corresponding probabilities.
    """
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    label_names = load_label_names(current_model)
    # Predict the class
    prediction = model.predict(image, normalize=True, batch_size=1)
    top_indices = np.argsort(prediction[0])[-5:][::-1]
    top_labels = []
    for i in top_indices:
        top_labels.append(label_names[i])
    top_probabilities = prediction[0][top_indices]
    for i in range(5):
        print(f"Class {top_labels[i]}: Probability {top_probabilities[i]}")
    ret_dict = {}
    for i in range(5):
        ret_dict[top_labels[i]] = top_probabilities[i]
    class_index = np.argmax(prediction, axis=1)

    return class_index, ret_dict


def gen_prod_figure(pred_dict, current_model):
    """
    Generate a bar chart figure showing the class probabilities for a given prediction dictionary.

    Parameters:
    pred_dict (dict): A dictionary containing the class names as keys and their corresponding probabilities as values.
    current_model (str): The name of the current model.

    Returns:
    fig (go.Figure): The generated bar chart figure.
    """
    if pred_dict == 0:
        classes = ['', '', '', '', '']
        probabilities = [0, 0, 0, 0, 0]

        fig = go.Figure()

        # Add bar trace with colors
        fig.add_trace(go.Bar(
            x=classes,
            y=probabilities,
        ))

        fig.update_layout(
            title_text=f'Class Probabilities for {current_model}',
            title_x=0.5,
            xaxis=dict(title='Classes'),
            yaxis=dict(range=[0, 1]),
        )

        return fig

    classes = list(pred_dict.keys())
    probabilities = list(pred_dict.values())
    colors = ['blue', 'green', 'red', 'purple', 'orange']

    fig = go.Figure()

    # Add bar trace with colors
    fig.add_trace(go.Bar(
        x=classes,
        y=probabilities,
        marker_color=colors,
        text=probabilities,
        textposition='auto',
    ))

    fig.update_layout(
        title_text=f'Class Probabilities for {current_model}',
        title_x=0.5,
        xaxis=dict(title='Classes'),
        yaxis=dict(title='Probability'),
    )

    return fig
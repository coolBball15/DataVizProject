import numpy as np
from keras.preprocessing.image import img_to_array
import plotly.graph_objects as go

from data_loader import load_label_names

def predict_image_class(model, image):
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    label_names = load_label_names('cifar10')
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

    return class_index,ret_dict


def gen_prod_figure(pred_dict):
    if pred_dict ==0:
        classes = ['', '', '', '', '']
        probabilities = [0, 0, 0, 0, 0] 

        fig = go.Figure()

        # Add bar trace with colors
        fig.add_trace(go.Bar(
            x=classes,
            y=probabilities,
        ))

        fig.update_layout(
            title='Class Probabilities for CIFAR-10',
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
        title='Class Probabilities for CIFAR-10',
        xaxis=dict(title='Classes'),
        yaxis=dict(title='Probability'),
    )

    return fig
import plotly.express as px
import numpy as np
from sklearn.decomposition import PCA
from PIL import Image
import base64
import io
from io import BytesIO
from lime import lime_image
from sklearn.manifold import TSNE
from tensorflow.keras.preprocessing import image    
from tensorflow.keras.applications.vgg16 import preprocess_input
from skimage.segmentation import mark_boundaries




def pca_decomposition(activations):
    """
    Perform PCA decomposition on the given activations.

    Parameters:
    activations (ndarray): The activations to be decomposed.

    Returns:
    ndarray: The PCA decomposition of the activations.
    """
    # Reshape the 4D array into a 2D matrix
    num_samples = activations.shape[0]
    num_features = np.prod(activations.shape[1:])
    data_2d = activations.reshape(num_samples, num_features)

    # Standardize the data (optional but recommended for PCA)
    mean = np.mean(data_2d, axis=0)
    std = np.std(data_2d, axis=0)
    standardized_data = (data_2d - mean) / std

    pca = PCA(n_components=2)
    pca_transform = pca.fit_transform(standardized_data)
    return pca_transform

def tsne_decomposition(activations):
    """
    Perform t-SNE decomposition on the given activations.

    Parameters:
    activations (ndarray): The activations to be decomposed.

    Returns:
    ndarray: The t-SNE transformed data.
    """
    # Reshape the 4D array into a 2D matrix
    num_samples = activations.shape[0]
    num_features = np.prod(activations.shape[1:])
    data_2d = activations.reshape(num_samples, num_features)

    # Standardize the data (optional but recommended for PCA)
    mean = np.mean(data_2d, axis=0)
    std = np.std(data_2d, axis=0)
    standardized_data = (data_2d - mean) / std

    tsne = TSNE(n_components=2)
    tsne_transform = tsne.fit_transform(standardized_data)
    return tsne_transform

def plot(pca_data, labels):
    """
    Plots the PCA data with labeled points.
    
    Parameters:
    - pca_data (numpy.ndarray): The PCA data to be plotted.
    - labels (numpy.ndarray): The labels for each data point.
    """

    fig = px.scatter(
        x=pca_data[:, 0],
        y=pca_data[:, 1], 
        color=labels,
        hover_name = [i for i in range(len(pca_data))]
        )
    return fig

def get_image(image_data):
    """
    Converts a NumPy array to a base64-encoded image in PNG format.

    Parameters:
    image_data (numpy.ndarray): The input image data as a NumPy array.

    Returns:
    str: The base64-encoded image in PNG format.
    """
    selected_img = Image.fromarray((image_data).astype('uint8'))
    selected_img_byte_array = io.BytesIO()
    selected_img.save(selected_img_byte_array, format='PNG')
    selected_img_base64 = base64.b64encode(selected_img_byte_array.getvalue()).decode('utf-8')

    return f'data:image/png;base64,{selected_img_base64}'

def get_image_LIME(model, img):
    """
    Generates a LIME explanation for an image prediction using the given model.

    Parameters:
    - model: The trained model used for prediction.
    - img: The input image.

    Returns:
    - img_boundry: The visualized LIME explanation for the image.
    """
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    best = np.argmax(predictions[0])

    # Create a LIME explainer
    explainer = lime_image.LimeImageExplainer()

    # Define a predict function for the model
    def predict_function(images):
        images = preprocess_input(images)
        return model.predict(images, normalize=True, batch_size=1)

    # Explain the prediction for the image using LIME
    explanation = explainer.explain_instance(img_array[0], predict_function, top_labels=3, num_features=5)

    # Visualize the LIME explanation
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
    img_boundry = mark_boundaries(temp / 2 + 0.5, mask)

    return img_boundry

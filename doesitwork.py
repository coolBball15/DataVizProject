import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from keras.models import Sequential
from lime import lime_image
from skimage.segmentation import mark_boundaries

# Load a pre-trained VGG16 model (you can replace this with your CIFAR-100 model)
model = Sequential()
model.load_weights('data/cifar100vgg.h5')

# Load and preprocess an example image from CIFAR-100
img_path = 'poes.jpg'  # Replace with the path to your image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Make a prediction using the VGG16 model
predictions = model.predict(img_array)
decoded_predictions = decode_predictions(predictions, top=3)[0]
print("Predictions:", decoded_predictions)

# Create a LIME explainer
explainer = lime_image.LimeImageExplainer()

# Define a predict function for the model
def predict_function(images):
    images = preprocess_input(images)
    return model.predict(images)

# Explain the prediction for the image using LIME
explanation = explainer.explain_instance(img_array[0], predict_function, top_labels=3, num_features=5)

# Visualize the LIME explanation
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
img_boundry = mark_boundaries(temp / 2 + 0.5, mask)

# Plot the original image and the LIME explanation side by side
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(img_boundry)
plt.title("LIME Explanation")

plt.show()
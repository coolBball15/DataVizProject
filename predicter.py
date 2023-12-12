import numpy as np
from keras.preprocessing.image import img_to_array

def predict_image_class(model, image):
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # Predict the class
    prediction = model.predict(image, normalize=True, batch_size=1)

    # Assuming the prediction is a one-hot encoded vector, get the class index
    class_index = np.argmax(prediction, axis=1)

    return class_index
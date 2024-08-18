# from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image
def preprocess_image(path):
    # img = load_img(path, target_size=(224,224))
    # img = img_to_array(img)
    # img = np.expand_dims(img, axis=0)
    # img = img / 224
    
    # return img
    
    img = Image.open(path).convert('RGB') 
    # Resize the image
    img = img.resize((224, 224))
    # Convert the image to an array
    img_array = np.array(img)
    # Normalize pixel values to the range [0, 1]
    img_array = img_array / 255.0
    # Add a batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array, img
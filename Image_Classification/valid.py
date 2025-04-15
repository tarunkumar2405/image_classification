import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import json

# Load the model
with open('ajith_vijay_classifier.json', 'r') as json_file:
    model_json = json_file.read()

model = tf.keras.models.model_from_json(model_json)
model.load_weights('ajith_vijay_classifier.weights.h5')  # Correct filename
print("Model loaded successfully.")

# Function to classify images
def classify_images(test_dir):
    for img_name in os.listdir(test_dir):
        img_path = os.path.join(test_dir, img_name)
        if not img_name.lower().endswith(('png', 'jpg', 'jpeg')):
            continue
        
        img = image.load_img(img_path, target_size=(150, 150))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array)
        if prediction[0][0] > 0.5:
            print(f"{img_name}: Thalapathy Vijay")
        else:
            print(f"{img_name}: Thala Ajith")

# Path to test images
test_dir = 'dataset/test'

if os.path.exists(test_dir):
    print("Validating images in test folder...")
    classify_images(test_dir)
else:
    print(f"Test folder '{test_dir}' not found.")

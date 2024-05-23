from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
# Load the trained model
model = load_model("digitSignLanguage.h5")
# Load the image you want to predict
image_path = "9.jpeg"  # Replace with the path to your image
new_image = Image.open(image_path)
# Ensure the image has 3 color channels (RGB)
if new_image.mode != "RGB":
    new_image = new_image.convert("RGB")
# Preprocess the image
resized_image = new_image.resize((32, 32))
input_image = np.array(resized_image) / 255.0
input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension
# Make predictions
predictions = model.predict(input_image)
predicted_class_index = np.argmax(predictions)
print(predicted_class_index)
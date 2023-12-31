import streamlit as st
from PIL import Image
import tensorflow as tf
configuration={'batchsize':32,'img_size':256,
               'learningrate':0.001,
               'n_epoches':12,
               "num_classes":12,
               'droupoutrate':0.0,
               'regularization_rate':0.0 ,
               'num_filters':6,
               "kernelsize":3,
               "n_strides":1,
               'poolsize':2,
               'N_DENSE_1':100,
               'N_DENSE_2':10,
}
# Load the model for testing
loaded_model = tf.keras.models.load_model('pests_detection_model.h5')

# Define the class names (replace with your actual class names)
class_names = ["ants", 'bees', 'beetle', 'caterpillar', 'earthworms', 'earwig', 'grasshopper', 'moth', 'slug', 'snail',
               'wasp', 'weevil']

# Function to preprocess the image and make predictions
def predict_with_model(image):
    # Load and preprocess the image
    processed_image = image.resize((configuration["img_size"], configuration["img_size"]))
    processed_image = tf.keras.preprocessing.image.img_to_array(processed_image)
    processed_image = tf.expand_dims(processed_image, axis=0)

    # Make predictions
    predictions = loaded_model(processed_image)

    # Get the predicted class
    predicted_class = class_names[tf.argmax(predictions, axis=-1).numpy()[0]]

    return predicted_class.upper()

def main():
    st.title("Pests Detection App")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        if st.button("Predict"):
            result = predict_with_model(image)
            st.success(f"Predicted Class: {result}")

if __name__ == "__main__":
    main()

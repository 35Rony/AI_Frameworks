import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import cv2
from streamlit_drawable_canvas import st_canvas  # pip install streamlit-drawable-canvas

# --- Model building and training ---

@st.cache_resource(show_spinner=False)
def load_or_train_model():
    try:
        # Try to load existing saved model
        model = tf.keras.models.load_model('mnist_cnn_model.h5')
    except:
        # Load MNIST dataset
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

        # Preprocess data
        train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
        test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

        # Build CNN model
        model = tf.keras.models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])

        # Compile model
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        # Train model for 5 epochs
        model.fit(train_images, train_labels, epochs=5, validation_split=0.1, batch_size=64)

        # Save model
        model.save('mnist_cnn_model.h5')

    return model

model = load_or_train_model()

# --- Streamlit app ---

st.title("MNIST CNN Digit Classifier")

st.write("Draw a digit (0-9) on the canvas below:")

canvas = st_canvas(
    stroke_width=15,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas.image_data is not None:
    img = canvas.image_data[:, :, 0]  # grayscale channel from RGBA
    img = cv2.resize(img, (28, 28))
    img = 255 - img  # invert colors: white digit on black
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1).astype('float32')

    prediction = model.predict(img)
    predicted_digit = np.argmax(prediction)

    st.write(f"Predicted Digit: {predicted_digit}")
    st.image(img.reshape(28, 28), width=100, caption="Preprocessed Input")

st.write("Note: The model trains once if not already saved, then loads for fast predictions.")

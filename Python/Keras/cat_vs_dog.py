"""
Convolutional Neural Network (CNN) for Image Classification using Keras
This program builds, compiles, and trains a CNN to classify images of cats and dogs.
It uses TensorFlow/Keras's modern image loading API and expects the following directory structure:

- dataset/
    - training_set/
        - training_set/
            - cats/
            - dogs/
    - test_set/
        - test_set/
            - cats/
            - dogs/
"""

# Import necessary Keras modules for building the CNN
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten

############################################################
# 1. Build the CNN model
############################################################
# Initialising the CNN as a Sequential model
classifier = Sequential()

# Step 1 - First Convolutional Layer
# Applies 32 filters of size 3x3 to the input image (64x64 RGB)
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation="relu"))

# Step 2 - Second Convolutional Layer and Pooling
# Adds another convolutional layer and a max pooling layer to reduce spatial dimensions
classifier.add(Conv2D(32, (3, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flattening
# Converts the 2D feature maps to a 1D feature vector
classifier.add(Flatten())

# Step 4 - Fully Connected Layers (Dense)
# Adds a hidden layer and an output layer for binary classification
classifier.add(Dense(units=128, activation="relu"))
classifier.add(Dense(units=1, activation="sigmoid"))

# Step 5 - Compile the CNN
# Uses Adam optimizer and binary crossentropy loss for binary classification
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


############################################################
# 2. Load and preprocess the image data
############################################################
# Uses Keras's image_dataset_from_directory to load images from folders
# Images are resized to 64x64 and labels are inferred from folder names
from keras.utils import image_dataset_from_directory

# Load training images (cats and dogs)
training_set = image_dataset_from_directory(
    "dataset/training_set/training_set",  # Path to training images
    image_size=(64, 64),                   # Resize images to 64x64
    batch_size=32,                         # Number of images per batch
    interpolation="bilinear",             # Interpolation method for resizing
    label_mode="binary",                  # Binary labels (cats=0, dogs=1)
)

# Load test images (cats and dogs)
test_set = image_dataset_from_directory(
    "dataset/test_set/test_set",          # Path to test images
    image_size=(64, 64),
    batch_size=32,
    interpolation="bilinear",
    label_mode="binary",
)

############################################################
# 3. Train the CNN model
############################################################
# Fits the model to the training data and validates on the test data
classifier.fit(
    training_set,
    epochs=25,                # Number of training epochs
    validation_data=test_set, # Validation data for monitoring accuracy
)


import numpy as np
from keras.preprocessing import image
# Load and preprocess a single image for prediction
test_image = image.load_img("dataset/single_prediction/cat_or_dog_1.jpg", target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)  # Add batch dimension
result = classifier.predict(test_image)  # Predict the class of the image
training_set.class_names  # Get class names to interpret the result
if result[0][0] >= 0.5:
    prediction = "dog"
else:
    prediction = "cat"
print(f"The predicted class is: {prediction}")
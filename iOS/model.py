# Import libraries
import numpy as np
import pandas as pd
import tensorflow.compat.v2 as tf  # Change this line

tf.enable_v2_behavior()  # Add this line
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define constants
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NUM_CLASSES = len(EMOTIONS)
IMAGE_SIZE = (48, 48)
BATCH_SIZE = 32

# Create image data generator
datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

# Create train and validation generators
train_generator = datagen.flow_from_directory(
    directory="images/train",
    target_size=IMAGE_SIZE,
    color_mode="grayscale",
    classes=EMOTIONS,
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    subset="training"
)

validation_generator = datagen.flow_from_directory(
    directory="train",
    target_size=IMAGE_SIZE,
    color_mode="grayscale",
    classes=EMOTIONS,
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    subset="validation"
)

# Create model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES, activation="softmax"))

# Compile model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train model
model.fit(train_generator, epochs=10, validation_data=validation_generator)

# Save model
model.save("emotion_detector.h5")

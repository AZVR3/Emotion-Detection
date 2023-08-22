# Import libraries
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("windows/emotion_detector.h5")

# Define constants
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NUM_CLASSES = len(EMOTIONS)
IMAGE_SIZE = (48, 48)

# Create face detector
face_cascade = cv2.CascadeClassifier("windows/haarcascade_frontalface_default.xml")

# Create video capture
cap = cv2.VideoCapture(0)

# Loop until user presses q
while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Loop over the faces
    for (x, y, w, h) in faces:
        # Crop the face region
        face = gray[y:y+h, x:x+w]

        # Resize the face to match the model input size
        face = cv2.resize(face, IMAGE_SIZE)

        # Normalize the face pixels
        face = face / 255.0

        # Reshape the face to match the model input shape
        face = face.reshape(1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)

        # Predict the emotion of the face
        emotion = model.predict(face)

        # Get the index of the highest probability emotion
        emotion_index = np.argmax(emotion)

        # Get the label of the emotion
        emotion_label = EMOTIONS[emotion_index]

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Put the emotion label above the face
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show the frame with the faces and emotions
    cv2.imshow("Emotion Detector", frame)

    # Wait for user input
    key = cv2.waitKey(1) & 0xFF

    # If user presses q, exit the loop
    if key == ord("q"):
        break

# Release video capture and destroy windows
cap.release()
cv2.destroyAllWindows()

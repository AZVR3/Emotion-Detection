# Import the deepface library
from deepface import DeepFace

# Define the emotion labels
emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "surprise"]

# Define the training data path
train_data_path = "../images/train"

# Train the emotion recognition model using the deepface library
model = DeepFace.build_model("Emotion")

# Print the model summary
model.summary()

# Compile the model with loss function and optimizer
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Fit the model on the training data
model.fit(train_data_path, emotion_labels, epochs=10, batch_size=32)

# Save the model
model.save("emotion_model.h5")

# Import the opencv library for webcam access
import cv2

# Create a video capture object to access the webcam
video_capture = cv2.VideoCapture(0)

# Loop until the user presses 'q' key
while True:
    # Capture a frame from the webcam
    _, frame = video_capture.read()

    # Resize the frame to 224x224 pixels for the model input
    resized_frame = cv2.resize(frame, (224, 224))

    # Predict the emotion of the face in the frame using the deepface library
    emotion_prediction = DeepFace.analyze(resized_frame, actions=["emotion"], models={"emotion": model})

    # Get the dominant emotion and its confidence score
    dominant_emotion = emotion_prediction["dominant_emotion"]
    emotion_score = emotion_prediction["emotion"][dominant_emotion]

    # Display the emotion and score on the frame
    cv2.putText(frame, f"{dominant_emotion}: {emotion_score:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the frame on a window
    cv2.imshow("Emotion Detection", frame)

    # Check if the user presses 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and close all windows
video_capture.release()
cv2.destroyAllWindows()

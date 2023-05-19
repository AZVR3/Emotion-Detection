# Import opencv library
import cv2

# Create a video capture object
cap = cv2.VideoCapture(0)

# Check if the camera is opened
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

# Create a background subtractor object
bg_sub = cv2.createBackgroundSubtractorMOG2()

# Create a kernel for morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Loop until the user presses 'q' key
while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Check if the frame is valid
    if not ret:
        print("Error: Cannot read frame")
        break

    # Apply background subtraction to get the foreground mask
    fg_mask = bg_sub.apply(frame)

    # Blur the foreground mask to reduce noise
    fg_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0)

    # Threshold the foreground mask to get a binary image
    _, fg_mask = cv2.threshold(fg_mask, 25, 255, cv2.THRESH_BINARY)

    # Apply morphological operations to fill gaps and remove small blobs
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    # Find the contours of the foreground mask
    contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours on the frame
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    # Check if any contour is large enough to indicate movement
    for c in contours:
        if cv2.contourArea(c) > 1000:
            print("Movement detected")
            break

    # Display the frame and the foreground mask in two windows
    cv2.imshow("Camera", frame)
    cv2.imshow("Foreground", fg_mask)

    # Wait for 1 millisecond and check for user input
    key = cv2.waitKey(1)
    if key == ord('q'):
        print("Exiting...")
        break

# Release the camera and destroy the windows
cap.release()
cv2.destroyAllWindows()

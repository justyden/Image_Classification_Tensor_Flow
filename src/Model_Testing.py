import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load your trained model
model = tf.keras.models.load_model('src/model')

# Define the desired size of the video feed
VIDEO_FEED_SIZE = (640, 480)

# Define the size of images expected by the model
IMG_SIZE = (160, 160)

# Access the camera (0 represents the default camera, change it if you have multiple cameras)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Resize the frame to the desired size for display
    frame_display = cv2.resize(frame, VIDEO_FEED_SIZE)

    # Preprocess the frame to match the input size expected by the model
    frame_resized = cv2.resize(frame, IMG_SIZE)
    img_array = image.img_to_array(frame_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # Make predictions using the model
    predictions = model.predict(img_array)

     # Get the predicted class
    predicted_class = np.argmax(predictions)
    if predicted_class == 0:
        predicted_class = "Lid Misaligned"
    elif predicted_class == 1:
        predicted_class = "Lid Missing"
    elif predicted_class == 2:
        predicted_class = "Lid Open"
    elif predicted_class == 3:
        predicted_class = "Lid Properly Placed"

    print("Raw Predictions:", predictions)

    # Display the frame with the predicted class
    cv2.putText(frame_display, f'Predicted Class: {predicted_class}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the resized frame
    cv2.imshow('Camera Feed', frame_display)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV window
cap.release()
cv2.destroyAllWindows()


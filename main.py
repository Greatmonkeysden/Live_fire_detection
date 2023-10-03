import cv2
import numpy as np
import tensorflow as tf

# Funtion to read image and give desired output with image
def pred_and_plot(model, frame, class_names):
    """
    Makes a prediction on the input frame with a trained model and returns the predicted class.
    """
    # Preprocess the frame for model input (resize and normalize pixel values)
    img_shape = 150
    img = cv2.resize(frame, (img_shape, img_shape))
    img = img / 255.0  # Normalize pixel values to [0, 1]

    # Ensure that the frame has 3 color channels (BGR)
    if img.shape[-1] != 3:
        return None  # Skip frames that don't have 3 channels

    # Make a prediction
    pred = model.predict(tf.expand_dims(img, axis=0))

    if len(pred[0]) > 1:  # Check for multi-class
        pred_class = class_names[pred.argmax()]  # If more than one output, take the max
    else:
        pred_class = class_names[int(tf.round(pred)[0][0])]  # If only one output, round

    return pred_class

# Load the pre-trained fire detection model
model_path = "C:\\Users\\HP\\Downloads\\Fire_detection_model.h5"
fire_detection_model = tf.keras.models.load_model(model_path)

# Open the default camera (camera index 0)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Check if the frame is valid
    if not ret or frame is None:
        continue

    class_names = ['Not-fire', 'Fire']

    # Reading the input and checking the output
    predicted_class = pred_and_plot(fire_detection_model, frame, class_names)

    if predicted_class:
        # Overlay the result on the frame
        cv2.putText(frame, predicted_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Convert the frame to the correct format for display
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display the processed frame with the result
    cv2.imshow('Fire Detection', frame_rgb)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

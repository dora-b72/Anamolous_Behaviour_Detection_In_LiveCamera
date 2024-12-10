import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your model
model = load_model("C:\\Users\\bhava\\.spyder-py3\\pose_classifier_model.h5")  # Update this path as needed

# Print model summary to understand input shape
model.summary()

# Initialize the camera
cap = cv2.VideoCapture(0)

# Define the threshold for classification
threshold = 0.5  # You can adjust this threshold as needed

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to the input size of the model (62, 62)
    resized_frame = cv2.resize(frame, (62, 62))  # Update the size to match your model's input
    # Normalize the frame
    preprocessed_frame = resized_frame / 255.0  # Normalize to [0, 1]
    preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)  # Add batch dimension

    # Check the shape of preprocessed_frame
    print(f"Shape of preprocessed_frame: {preprocessed_frame.shape}")

    # Make a prediction
    predictions = model.predict(preprocessed_frame)

    # Get the prediction probability
    prediction_probability = predictions[0][0]
    
    # Determine label based on the prediction and threshold
    if prediction_probability > threshold:
        label = 'Abnormal'
    else:
        label = 'Normal'

    # Display the label and prediction probability on the frame
    cv2.putText(frame, f'Label: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Probability: {prediction_probability:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Live Pose Classification', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

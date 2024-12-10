import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# Load your data
def load_data(data_dir):
    normal_data = []  # List to store normal poses
    abnormal_data = []  # List to store abnormal poses
    labels = []

    for label in ['normal', 'abnormal']:
        path = os.path.join(data_dir, label)
        if not os.path.exists(path):
            print(f"Warning: {path} does not exist.")
            continue
        for file in os.listdir(path):
            img = keras.preprocessing.image.load_img(os.path.join(path, file), target_size=(64, 64))
            img_array = keras.preprocessing.image.img_to_array(img)
            if label == 'normal':
                normal_data.append(img_array)
                labels.append(0)  # Label for normal
            else:
                abnormal_data.append(img_array)
                labels.append(1)  # Label for abnormal

    # Convert lists to NumPy arrays
    X = np.array(normal_data + abnormal_data)
    y = np.array(labels)

    return X, y

# Prepare your dataset
data_directory = r'C:\Users\bhava\.spyder-py3\dataset'  # Use raw string to avoid issues with backslashes
X, y = load_data(data_directory)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize your data
X_train = X_train.astype('float32') / 255.0
X_val = X_val.astype('float32') / 255.0

# Define your model
model = keras.Sequential([
    layers.Input(shape=(64, 64, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification: normal or abnormal
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Save the model
model.save('pose_classifier_model.h5')

print("Model training complete and saved.")

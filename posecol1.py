import cv2
import mediapipe as mp
import os

# Initialize MediaPipe pose detector
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Function to create directories if they don't exist
def create_dirs():
    if not os.path.exists('dataset/normal'):
        os.makedirs('dataset/normal')
    if not os.path.exists('dataset/abnormal'):
        os.makedirs('dataset/abnormal')

def collect_pose_data():
    create_dirs()  # Create folders if they don't exist
    cap = cv2.VideoCapture(0)
    label = 'normal'  # Default label
    last_label = None  # Track the previous label to prevent duplicate print messages

    img_count = {'normal': 0, 'abnormal': 0}  # Track number of images saved for each category

    print("Press 'n' for normal poses, 'a' for abnormal poses, 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the image to RGB for processing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # Draw pose landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Save the frame to the appropriate folder
            img_name = f'dataset/{label}/image_{img_count[label]}.jpg'
            cv2.imwrite(img_name, frame)

            # Update image count
            img_count[label] += 1

        # Display instructions and the current label on the frame
        cv2.putText(frame, f"Collecting {label} poses", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'n' for normal, 'a' for abnormal, 'q' to quit", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow('Pose Data Collection', frame)

        # Check for user input to switch labels or quit
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            label = 'normal'
        elif key == ord('a'):
            label = 'abnormal'

        # Only print when the label is changed
        if label != last_label:
            print(f"Switched to collecting {label} poses...")
            last_label = label

    cap.release()
    cv2.destroyAllWindows()
    print("Finished collecting pose data.")

# Example usage
if __name__ == '__main__':
    collect_pose_data()  # Collect both normal and abnormal poses

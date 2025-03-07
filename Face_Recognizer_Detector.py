import cv2
import numpy as np
import os

# Load pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Initialize the face recognizer (LBPH)
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load and label known images for simple recognition
known_faces = {
    "Person1": cv2.imread("C:\\Users\\deepak\\Documents\\cv\\pp.JPG", cv2.IMREAD_GRAYSCALE),
    "Person2": cv2.imread("C:\\Users\\deepak\\Documents\\cv\\mypic.jpg", cv2.IMREAD_GRAYSCALE)
}

# Resize and preprocess known images
processed_known_faces = {}
face_labels = []
label_names = []

for name, face in known_faces.items():
    if face is None:
        print(f"Error: Image for {name} could not be loaded. Skipping...")
        continue
    face = cv2.resize(face, (100, 100))  # Resize for consistency
    processed_known_faces[name] = face
    face_labels.append(0)  # Assigning a label (0 for Person1)
    label_names.append(name)

# Train the recognizer on known faces
recognizer.train(list(processed_known_faces.values()), np.array(face_labels))

# Create 'faces' folder if it doesn't exist
if not os.path.exists("faces"):
    os.makedirs("faces")

# Initialize webcam
video_capture = cv2.VideoCapture(0)

def recognize_face(detected_face):
    """
    Simple face recognition using LBPHFaceRecognizer.
    """
    label, confidence = recognizer.predict(detected_face)  # Predict the face label
    if confidence < 100:  # Set a threshold for better match
        return label_names[label]  # Return the name of the recognized person
    else:
        return "Unknown"  # If no match found

while True:
    # Capture video frame
    ret, frame = video_capture.read()

    # Convert to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for (x, y, w, h) in faces:
        detected_face = gray_frame[y:y + h, x:x + w]  # Crop the face region
        name = recognize_face(detected_face)  # Recognize the face

        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # If the face is unknown, allow registration
        if name == "Unknown":
            cv2.putText(frame, "Press 'r' to register face", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            if cv2.waitKey(1) & 0xFF == ord('r'):
                user_name = input("Enter your name: ")  # Prompt for a name in the console
                new_face_resized = cv2.resize(detected_face, (100, 100))
                processed_known_faces[user_name] = new_face_resized
                label_names.append(user_name)  # Add new label
                face_labels.append(len(label_names) - 1)  # Assign new label ID
                recognizer.update([new_face_resized], np.array([len(label_names) - 1]))

                # Save the new face to the 'faces' folder
                cv2.imwrite(f"faces/{user_name}.jpg", detected_face)
                print(f"New face saved as faces/{user_name}.jpg")

        # Display the name label
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the video feed
    cv2.imshow("Real-Time Face Detection and Recognition", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close windows
video_capture.release()
cv2.destroyAllWindows()

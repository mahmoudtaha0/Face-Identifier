import cv2
import face_recognition
import os

# Load images from the persons folder
known_faces = []
known_identifiers = []
for filename in os.listdir('Identifiers'):
    if filename.endswith(".jpg") or filename.endswith(".png")or filename.endswith(".jpeg"):
        image = face_recognition.load_image_file(f'Identifiers/{filename}')
        encoding = face_recognition.face_encodings(image)[0]
        known_faces.append(encoding)
        # Remove file extension
        known_identifiers.append(filename[:-4])  

# Capture video from webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Find all the faces in the current frame of video
    face_locations = face_recognition.face_locations(frame)

    # Display the results
    for top, right, bottom, left in face_locations:
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Identify the face
        face_encoding = face_recognition.face_encodings(frame, [(top, right, bottom, left)])[0]
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        identifier = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_identifiers[first_match_index]

        # Display face identifier
        cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
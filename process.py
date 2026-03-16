import cv2
import mediapipe as mp
import face_recognition
import os

KNOWN_FACES_DIR = "known_faces"
known_encodings = []
known_names = []

print("Loading known faces from directory...")

if not os.path.exists(KNOWN_FACES_DIR):
    print(f"Error: Directory '{KNOWN_FACES_DIR}' not found!")
    exit()

for filename in os.listdir(KNOWN_FACES_DIR):
    path = os.path.join(KNOWN_FACES_DIR, filename)
    
    try:
        image = face_recognition.load_image_file(path)
        
        list_of_encodings = face_recognition.face_encodings(image)
        
        if len(list_of_encodings) > 0:
            known_encodings.append(list_of_encodings[0])
            known_names.append(os.path.splitext(filename)[0])
            print(f"Success: Encoded {filename}")
        else:
            print(f"Skipping: No face detected in {filename}")
            
    except Exception as e:
        print(f"Error loading {filename}: {e}")

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

img = cv2.imread('khang.jpg')
if img is None:
    print("Error: Target image 'khang.jpg' not found.")
    exit()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = face_detection.process(img_rgb)

if results.detections:
    ih, iw, _ = img.shape
    print(f"System: Found {len(results.detections)} face(s).")
    
    for detection in results.detections:
        bbox = detection.location_data.relative_bounding_box
        x = int(bbox.xmin * iw)
        y = int(bbox.ymin * ih)
        w = int(bbox.width * iw)
        h = int(bbox.height * ih)
        
        x, y = max(0, x), max(0, y)
        
        face_locations = [(y, x + w, y + h, x)]
        current_encodings = face_recognition.face_encodings(img_rgb, face_locations)
        
        name = "Unknown"
        if len(current_encodings) > 0:
            matches = face_recognition.compare_faces(known_encodings, current_encodings[0])
            if True in matches:
                first_match_index = matches.index(True)
                name = known_names[first_match_index]

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        print(f"Result: {name} identified at x={x}, y={y}")

cv2.imshow('System...', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
import face_recognition
import pyodbc
import os
import sys
import io
from deepface import DeepFace

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

CONN_STR = "Driver={SQL Server};Server=localhost\SQLEXPRESS;Database=IdentityFinder;Trusted_Connection=yes;"
conn = pyodbc.connect(CONN_STR)
cursor = conn.cursor()

KNOWN_DIR = "known_faces"

if not os.path.exists(KNOWN_DIR):
    print(f"Error: Folder '{KNOWN_DIR}' not found!")
    exit()

print(f"System: Scanning identities and detecting gender from '{KNOWN_DIR}'...")

for filename in os.listdir(KNOWN_DIR):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(KNOWN_DIR, filename)
        user_name = os.path.splitext(filename)[0]

        image = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(image)

        if encodings:
            encoding_str = ",".join(map(str, encodings[0]))

            try:
                analysis = DeepFace.analyze(img_path=img_path, actions=['gender'], enforce_detection=False)
                if isinstance(analysis, list):
                    gender = analysis[0]['dominant_gender']
                else:
                    gender = analysis['dominant_gender']
            except Exception as e:
                print(f"Gender detection failed for {user_name}: {e}")
                gender = "Unknown"

            cursor.execute("SELECT UserID FROM Users WHERE UserName = ?", (user_name,))
            existing_user = cursor.fetchone()

            if existing_user:
                cursor.execute("""
                    UPDATE Users 
                    SET FaceData = ?, Gender = ? 
                    WHERE UserName = ?
                """, (encoding_str, gender, user_name))
                print(f"Updated Identity: {user_name} ({gender})")
            else:
                cursor.execute("""
                    INSERT INTO Users (UserName, FaceData, Gender) 
                    VALUES (?, ?, ?)
                """, (user_name, encoding_str, gender))
                print(f"Registered New Identity: {user_name} ({gender})")
            
            conn.commit()
        else:
            print(f"Warning: No face detected in {filename}")

print("--- [FINISH] Users table is now synchronized with Gender data ---")
conn.close()
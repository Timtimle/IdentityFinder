import os
import sys
import io
import face_recognition
import pyodbc
import numpy as np
from deepface import DeepFace

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

CONN_STR = "Driver={SQL Server};Server=localhost\\SQLEXPRESS;Database=IdentityFinder;Trusted_Connection=yes;"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSON_KNOWN_PATH = os.path.join(BASE_DIR, "known_faces", "person")

def sync_person_master_data():
    if not os.path.exists(PERSON_KNOWN_PATH):
        print(f"Error: Master directory not found at {PERSON_KNOWN_PATH}")
        return

    try:
        conn = pyodbc.connect(CONN_STR)
        cursor = conn.cursor()
        print("=" * 95)
        print(f"{'CATEGORY':<10} | {'USER NAME':<25} | {'IMG COUNT':<10} | {'STATUS'}")
        print("-" * 95)
    except Exception as e:
        print(f"SQL Connection Error: {e}")
        return

    for sub_name in os.listdir(PERSON_KNOWN_PATH):
        sub_path = os.path.join(PERSON_KNOWN_PATH, sub_name)
        if not os.path.isdir(sub_path): continue

        db_user_name = sub_name.lower()

        cursor.execute("SELECT UserName FROM Users WHERE UserName = ?", (db_user_name,))
        if cursor.fetchone():
            print(f"{'Person':<10} | {sub_name:<25} | --         | SKIPPED (Exists)")
            continue

        all_encodings = []
        img_files = [f for f in os.listdir(sub_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for filename in img_files:
            img_path = os.path.join(sub_path, filename)
            try:
                dets = DeepFace.extract_faces(img_path=img_path, detector_backend='retinaface', enforce_detection=False)
                
                if dets and dets[0]['confidence'] > 0.4:
                    best_face = max(dets, key=lambda x: x['facial_area']['w'] * x['facial_area']['h'])
                    r = best_face['facial_area']
                    face_loc = [(r['y'], r['x'] + r['w'], r['y'] + r['h'], r['x'])]
                    
                    image = face_recognition.load_image_file(img_path)
                    encs = face_recognition.face_encodings(image, face_loc, num_jitters=20, model="large")
                    
                    if encs:
                        all_encodings.append(encs[0])
            except Exception:
                continue

        if all_encodings:
            avg_enc = np.mean(all_encodings, axis=0)
            enc_str = ",".join(map(str, avg_enc))

            cursor.execute("INSERT INTO Users (UserName, FaceData, Gender) VALUES (?, ?, ?)", 
                           (db_user_name, enc_str, "Human"))
            conn.commit()
            print(f"{'Person':<10} | {sub_name:<25} | {len(all_encodings):<10} | SUCCESS")
        else:
            print(f"{'Person':<10} | {sub_name:<25} | 0          | FAILED")

    conn.close()
    print("=" * 95)
    print("FINISH: Master synchronization for 'person' folder complete.")

if __name__ == "__main__":
    sync_person_master_data()
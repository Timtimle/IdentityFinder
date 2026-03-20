import os
import sys
import io
import pyodbc
import numpy as np
import cv2
from deepface import DeepFace

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

CONN_STR = "Driver={SQL Server};Server=localhost\\SQLEXPRESS;Database=IdentityFinder;Trusted_Connection=yes;"
BASE_DIR = os.getcwd()
PERSON_KNOWN_PATH = os.path.join(BASE_DIR, "knowns_faces_1", "person")

def sync_person_master_data():
    if not os.path.exists(PERSON_KNOWN_PATH):
        print(f"ERROR: Directory not found: {PERSON_KNOWN_PATH}")
        return

    try:
        conn = pyodbc.connect(CONN_STR)
        cursor = conn.cursor()
        print("=" * 130)
        print(f"{'USER NAME':<25} | {'GENDER':<12} | {'VECTORS':<10} | {'ACTION':<15} | {'STATUS'}")
        print("-" * 130)
    except Exception as e:
        print(f"SQL Connection Error: {e}")
        return

    for sub_name in os.listdir(PERSON_KNOWN_PATH):
        sub_path = os.path.join(PERSON_KNOWN_PATH, sub_name)
        if not os.path.isdir(sub_path): continue

        db_user_name = sub_name.lower()
        img_files = [f for f in os.listdir(sub_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not img_files: continue

        cursor.execute("SELECT UserName FROM Users WHERE UserName = ?", (db_user_name,))
        if cursor.fetchone():
            print(f"{sub_name:<25} | {'--':<12} | {len(img_files):<10} | {'ALREADY SYNCED':<15} | SKIPPED")
            continue

        all_enc_strings = [] 
        all_genders = [] 

        for filename in img_files:
            img_path = os.path.join(sub_path, filename)
            try:
                img = cv2.imread(img_path)
                if img is None: continue
                h, w = img.shape[:2]
                center_x, center_y = w / 2, h / 2

                results = DeepFace.analyze(
                    img_path=img_path, actions=['gender'], 
                    detector_backend='retinaface', enforce_detection=True, align=True, silent=True
                )

                embeddings = DeepFace.represent(
                    img_path=img_path, model_name="Facenet512", 
                    detector_backend='retinaface', enforce_detection=True, align=True
                )

                if results and embeddings:
                    best_face = min(embeddings, key=lambda e: 
                        abs((e['facial_area']['x'] + e['facial_area']['w']/2) - center_x) + 
                        abs((e['facial_area']['y'] + e['facial_area']['h']/2) - center_y)
                    )
                    
                    if best_face.get('face_confidence', 0) >= 0.4:
                        vector = best_face['embedding']
                        all_enc_strings.append(",".join(map(str, vector)))
                        
                        all_genders.append(results[0]['dominant_gender'])

            except Exception as e:
                print(f"  WARN: {filename} skipped — {e}")
                continue

        if all_enc_strings:
            final_face_data = ";".join(all_enc_strings) 
            
            final_gender = max(set(all_genders), key=all_genders.count) if all_genders else "Unknown"
            
            abs_avatar_path = os.path.join(sub_path, img_files[0])
            rel_avatar_path = os.path.relpath(abs_avatar_path, BASE_DIR).replace("\\", "/")

            cursor.execute("""
                INSERT INTO Users (UserName, FaceData, Gender, UserAvatar) 
                VALUES (?, ?, ?, ?)
            """, (db_user_name, final_face_data, final_gender, rel_avatar_path))
            
            conn.commit()
            print(f"{sub_name:<25} | {final_gender:<12} | {len(all_enc_strings):<10} | {'INSERTED':<15} | SUCCESS")
        else:
            print(f"{sub_name:<25} | {'N/A':<12} | 0          | {'FAILED':<15} | ERROR")

    conn.close()
    print("-" * 130)
    print("[*] All Master Identities synchronized successfully.")

if __name__ == "__main__":
    sync_person_master_data()
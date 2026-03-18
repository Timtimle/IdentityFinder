import os
import sys
import io
import pyodbc
import numpy as np
from deepface import DeepFace

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

CONN_STR = "Driver={SQL Server};Server=localhost\\SQLEXPRESS;Database=IdentityFinder;Trusted_Connection=yes;"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSON_KNOWN_PATH = os.path.join(BASE_DIR, "knowns_faces_1", "person")

def sync_person_master_data():
    if not os.path.exists(PERSON_KNOWN_PATH):
        print(f"ERROR: Directory not found: {PERSON_KNOWN_PATH}")
        return

    try:
        conn = pyodbc.connect(CONN_STR)
        cursor = conn.cursor()
        print("=" * 130)
        print(f"{'USER NAME':<25} | {'GENDER':<12} | {'IMGS':<10} | {'ACTION':<15} | {'STATUS'}")
        print("-" * 130)
    except Exception as e:
        print(f"SQL Connection Error: {e}")
        return

    for sub_name in os.listdir(PERSON_KNOWN_PATH):
        sub_path = os.path.join(PERSON_KNOWN_PATH, sub_name)
        if not os.path.isdir(sub_path): continue

        db_user_name = sub_name.lower()
        img_files = [f for f in os.listdir(sub_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        current_count = len(img_files)
        
        if current_count == 0: continue

        cursor.execute("SELECT UserName FROM Users WHERE UserName = ?", (db_user_name,))
        if cursor.fetchone():
            print(f"{sub_name:<25} | {'--':<12} | {current_count:<10} | {'ALREADY SYNCED':<15} | SKIPPED")
            continue

        all_encodings = []
        all_genders = [] 

        for filename in img_files:
            img_path = os.path.join(sub_path, filename)
            try:
                analysis = DeepFace.analyze(img_path=img_path, actions=['gender'], 
                                           detector_backend='retinaface', enforce_detection=False, silent=True)
                if analysis:
                    all_genders.append(analysis[0]['dominant_gender'])
                    
                    # Facenet512 embedding
                    embeddings = DeepFace.represent(img_path=img_path, model_name="Facenet512", 
                                                   detector_backend='retinaface', enforce_detection=False)
                    if embeddings:
                        all_encodings.append(embeddings[0]['embedding'])
            except: continue

        if all_encodings:
            avg_enc = np.mean(all_encodings, axis=0)
            enc_str = ",".join(map(str, avg_enc))
            final_gender = max(set(all_genders), key=all_genders.count)
            
            abs_avatar_path = os.path.join(sub_path, img_files[0])
            rel_avatar_path = os.path.relpath(abs_avatar_path, BASE_DIR).replace("\\", "/")

            cursor.execute("""
                INSERT INTO Users (UserName, FaceData, Gender, UserAvatar) 
                VALUES (?, ?, ?, ?)
            """, (db_user_name, enc_str, final_gender, rel_avatar_path))
            
            conn.commit()
            print(f"{sub_name:<25} | {final_gender:<12} | {len(all_encodings):<10} | {'INSERTED':<15} | SUCCESS")
        else:
            print(f"{sub_name:<25} | {'N/A':<12} | 0          | {'FAILED':<15} | ERROR")

    conn.close()
    print("=" * 130)

if __name__ == "__main__":
    sync_person_master_data()
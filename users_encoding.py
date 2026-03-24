import os
import sys
import io
import logging
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

import pyodbc
import numpy as np
import cv2
from insightface.app import FaceAnalysis

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

CONN_STR          = "Driver={SQL Server};Server=localhost\\SQLEXPRESS;Database=IdentityFinder;Trusted_Connection=yes;"
BASE_DIR          = os.getcwd()
PERSON_KNOWN_PATH = os.path.join(BASE_DIR, "knowns_faces_1", "person")

app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

def setup_database(cursor):
    """ Initialize UserEmbeddings table with updated schema """
    cursor.execute("IF EXISTS (SELECT * FROM sysobjects WHERE name='UserEmbeddings' and xtype='U') DROP TABLE UserEmbeddings")
    cursor.execute("""
        CREATE TABLE UserEmbeddings (
            EmbeddingID INT IDENTITY(1,1) PRIMARY KEY,
            UserName NVARCHAR(255),
            FaceData VARCHAR(MAX),
            AngleHint VARCHAR(50)
        )
    """)

def sync_with_insightface():
    """ 
    Sync master data using InsightFace's 3D-aligned embeddings.
    This replaces the old DeepFace logic for superior accuracy on profile faces.
    """
    if not os.path.exists(PERSON_KNOWN_PATH):
        print(f"ERROR: Directory {PERSON_KNOWN_PATH} not found."); return

    try:
        conn = pyodbc.connect(CONN_STR)
        cursor = conn.cursor()
        setup_database(cursor)
        cursor.execute("DELETE FROM Users")
        conn.commit()
    except Exception as e:
        print(f"Database Error: {e}"); return

    print("=" * 115)
    print(f"{'INSIGHTFACE BUFFALO-L IDENTITY SYNC - OMNI-DIRECTIONAL MODE':^115}")
    print("=" * 115)

    for sub_name in sorted(os.listdir(PERSON_KNOWN_PATH)):
        sub_path = os.path.join(PERSON_KNOWN_PATH, sub_name)
        if not os.path.isdir(sub_path): continue

        db_user_name = sub_name.lower()
        img_files = [f for f in os.listdir(sub_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not img_files: continue

        embeddings_found = 0
        print(f"[*] Extracting high-precision features for: {sub_name.upper()}")

        for filename in img_files:
            img_path = os.path.join(sub_path, filename)
            img = cv2.imread(img_path)
            if img is None: continue

            faces = app.get(img)

            for face in faces:
                vec = face.normed_embedding
                vector_str = ",".join(map(str, vec))
                
                gender_str = "Man" if face.gender == 1 else "Woman"

                try:
                    cursor.execute(
                        "INSERT INTO UserEmbeddings (UserName, FaceData, AngleHint) VALUES (?, ?, ?)",
                        (db_user_name, vector_str, "insightface_feature")
                    )
                    embeddings_found += 1
                except: continue

        if embeddings_found > 0:
            rel_avatar = os.path.relpath(os.path.join(sub_path, img_files[0]), BASE_DIR).replace("\\", "/")
            cursor.execute(
                "INSERT INTO Users (UserName, FaceData, Gender, UserAvatar) VALUES (?, ?, ?, ?)",
                (db_user_name, 'INSIGHTFACE_ENROLLED', 'Confirmed', rel_avatar)
            )
            conn.commit()
            print(f"    [SUCCESS] Captured {embeddings_found} facial feature points.")
        else:
            print(f"    [FAILED] No faces detected in source images for {sub_name}.")

    conn.close()
    print("=" * 115 + "\nINSIGHTFACE SYNC COMPLETE. SYSTEM IS NOW AT MAXIMUM PRECISION.")

if __name__ == "__main__":
    sync_with_insightface()
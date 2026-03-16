import os
import sys
import io
import time
import torch
import face_recognition
import pyodbc
import numpy as np
from deepface import DeepFace

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ANIME_PATH = os.path.join(BASE_DIR, "anime_detection")
if ANIME_PATH not in sys.path: 
    sys.path.append(ANIME_PATH)

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox

CONN_STR = "Driver={SQL Server};Server=localhost\\SQLEXPRESS;Database=IdentityFinder;Trusted_Connection=yes;"
TEST_FOLDER = os.path.join(BASE_DIR, "test_100_celebA")
WEIGHTS = os.path.join(ANIME_PATH, "weights", "yolov5s_anime.pt")

device = torch.device('cpu')
anime_model = attempt_load(WEIGHTS, map_location=device)

def get_anime_locations(img0):
    img = letterbox(img0, 640)[0]
    img = torch.from_numpy(img.transpose(2, 0, 1)).to(device).float() / 255.0
    if img.ndimension() == 3: img = img.unsqueeze(0)
    
    with torch.no_grad():
        pred = anime_model(img, augment=False)[0]
    
    pred = non_max_suppression(pred, 0.2, 0.45) 
    
    locs = []
    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(int, xyxy)
                w, h = x2 - x1, y2 - y1
                locs.append({'region': [x1, y1, w, h]})
    return locs

def ultra_mass_encoding():
    try:
        conn = pyodbc.connect(CONN_STR)
        cursor = conn.cursor()
        print("-" * 90)
        print("MASS ENCODING START - HYBRID DETECTION STRATEGY")
        print("-" * 90)
    except Exception as e:
        print(f"Database Connection Error: {e}")
        return

    all_files = [f for f in os.listdir(TEST_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total = len(all_files)

    for index, filename in enumerate(all_files):
        cursor.execute("SELECT TOP 1 LabelID FROM ImageLabels WHERE FileName = ?", (filename,))
        if cursor.fetchone(): continue

        file_path = os.path.join(TEST_FOLDER, filename)
        
        anime_keywords = ["anime", "waifu", "vn", "kazusa", "haruki", "kurisu"]
        is_anime = any(kw in filename.lower() for kw in anime_keywords)

        try:
            start_time = time.time()
            image = face_recognition.load_image_file(file_path)
            faces_count = 0
            
            if is_anime:
                anime_faces = get_anime_locations(image)
                for item in anime_faces:
                    x, y, w, h = item['region']
                    cursor.execute("""
                        INSERT INTO ImageLabels (FileName, FilePath, LabelName, Gender, BoxX, BoxY, BoxW, BoxH, FaceData)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)
                    """, (filename, file_path, 'Anime_Character', 'Anime', x, y, w, h))
                faces_count = len(anime_faces)
            else:
                faces = DeepFace.extract_faces(img_path=file_path, detector_backend='retinaface', enforce_detection=False)
                for f in faces:
                    if f['confidence'] > 0.4:
                        r = f['facial_area']
                        encs = face_recognition.face_encodings(image, [(r['y'], r['x']+r['w'], r['y']+r['h'], r['x'])], model="large")
                        if encs:
                            enc_str = ",".join(map(str, encs[0]))
                            cursor.execute("""
                                INSERT INTO ImageLabels (FileName, FilePath, LabelName, Gender, BoxX, BoxY, BoxW, BoxH, FaceData)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (filename, file_path, 'Person', 'Human', r['x'], r['y'], r['w'], r['h'], enc_str))
                            faces_count += 1
            
            conn.commit()
            duration = time.time() - start_time
            print(f"[{index+1}/{total}] {filename} | Anime: {is_anime} | Faces: {faces_count} | {duration:.2f}s")
            
        except Exception as e:
            print(f"Processing Error at {filename}: {e}")

    conn.close()
    print("-" * 90)
    print("FINISHED: All files processed.")

if __name__ == "__main__":
    ultra_mass_encoding()
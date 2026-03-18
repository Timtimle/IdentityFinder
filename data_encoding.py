import os
import sys
import io
import time
import torch
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

# SQL & Folder Config
CONN_STR = "Driver={SQL Server};Server=localhost\\SQLEXPRESS;Database=IdentityFinder;Trusted_Connection=yes;"
TEST_FOLDER = os.path.join(BASE_DIR, "test_per")
WEIGHTS = os.path.join(ANIME_PATH, "weights", "yolov5s_anime.pt")

# Load Anime Model
device = torch.device('cpu')
anime_model = attempt_load(WEIGHTS, map_location=device)

def get_anime_locations(img0):
    """ Detects anime faces using YOLOv5 """
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
                locs.append({'region': [x1, y1, w, h], 'confidence': float(conf)})
    return locs

def ultra_mass_encoding():
    """ 
    Mass Encoding Engine:
    - Anime: YOLOv5 Detection
    - Human: RetinaFace + FaceNet 512 + Gender Analysis
    """
    try:
        conn = pyodbc.connect(CONN_STR)
        cursor = conn.cursor()
        print("-" * 110)
        print(f"{'MASS ENCODING START (FACENET-512 MODE)':^110}")
        print("-" * 110)
    except Exception as e:
        print(f"Database Connection Error: {e}")
        return

    all_files = [f for f in os.listdir(TEST_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total = len(all_files)

    for index, filename in enumerate(all_files):
        cursor.execute("SELECT TOP 1 LabelID FROM ImageLabels WHERE FileName = ?", (filename,))
        if cursor.fetchone(): 
            continue

        file_path = os.path.join(TEST_FOLDER, filename)
        
        anime_keywords = ["anime", "waifu", "vn", "kazusa", "haruki", "kurisu", "white_album"]
        is_anime = any(kw in filename.lower() for kw in anime_keywords)

        try:
            start_time = time.time()
            faces_count = 0
            
            if is_anime:
                import cv2
                image_cv = cv2.imread(file_path)
                anime_faces = get_anime_locations(image_cv)
                for item in anime_faces:
                    x, y, w, h = item['region']
                    conf = item['confidence']
                    cursor.execute("""
                        INSERT INTO ImageLabels (FileName, FilePath, LabelName, Gender, BoxX, BoxY, BoxW, BoxH, FaceData, Confidence)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
                    """, (filename, file_path, 'Anime_Character', 'Anime', x, y, w, h, conf))
                faces_count = len(anime_faces)
            
            else:
                results = DeepFace.analyze(
                    img_path=file_path, 
                    actions=['gender'], 
                    detector_backend='retinaface', 
                    enforce_detection=False,
                    silent=True
                )
                
                embeddings = DeepFace.represent(
                    img_path=file_path,
                    model_name="Facenet512",
                    detector_backend='retinaface',
                    enforce_detection=False
                )
                
                for i, f in enumerate(results):
                    if f['face_confidence'] > 0.4:
                        r = f['region']
                        gender_label = f['dominant_gender']
                        conf = f['face_confidence']
                        
                        if i < len(embeddings):
                            vector_512 = embeddings[i]['embedding']
                            enc_str = ",".join(map(str, vector_512))
                            
                            cursor.execute("""
                                INSERT INTO ImageLabels (FileName, FilePath, LabelName, Gender, BoxX, BoxY, BoxW, BoxH, FaceData, Confidence)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (filename, file_path, 'Person', gender_label, r['x'], r['y'], r['w'], r['h'], enc_str, conf))
                            faces_count += 1
            
            conn.commit()
            duration = time.time() - start_time
            print(f"[{index+1}/{total}] {filename[:25]:<25} | Faces: {faces_count} | {duration:.2f}s | OK")
            
        except Exception as e:
            print(f"Error at {filename}: {e}")

    conn.close()
    print("-" * 110)
    print("FINISHED: Database updated with FaceNet-512 Embeddings and Gender.")

if __name__ == "__main__":
    ultra_mass_encoding()
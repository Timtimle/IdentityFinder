import os
import sys
import io
import time
import torch
import pyodbc
import numpy as np
import cv2
import logging
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

from insightface.app import FaceAnalysis

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_DIR = os.getcwd()
ANIME_PATH = os.path.join(BASE_DIR, "anime_detection")
if ANIME_PATH not in sys.path: 
    sys.path.append(ANIME_PATH)

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox

CONN_STR = "Driver={SQL Server};Server=localhost\\SQLEXPRESS;Database=IdentityFinder;Trusted_Connection=yes;"
TEST_FOLDER = os.path.join(BASE_DIR, "test_per")
WEIGHTS = os.path.join(ANIME_PATH, "weights", "yolov5s_anime.pt")

app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

device = torch.device('cpu')
anime_model = None
if os.path.exists(WEIGHTS):
    anime_model = attempt_load(WEIGHTS, map_location=device)

def get_anime_locations(img0):
    if anime_model is None: return []
    img = letterbox(img0, 640)[0]
    img = torch.from_numpy(img.transpose(2, 0, 1)).to(device).float() / 255.0
    if img.ndimension() == 3: img = img.unsqueeze(0)
    with torch.no_grad():
        pred = anime_model(img, augment=False)[0]
    pred = non_max_suppression(pred, 0.4, 0.45) 
    locs = []
    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = map(int, xyxy)
                locs.append({'region': [x1, y1, x2 - x1, y2 - y1]})
    return locs

def process_test_images():
    try:
        conn = pyodbc.connect(CONN_STR)
        cursor = conn.cursor()
        print("-" * 110)
        print(f"{'INSIGHTFACE MASS ENCODING START (MAX PRECISION)':^110}")
        print("-" * 110)
    except Exception as e:
        print(f"Database Error: {e}"); return

    if not os.path.exists(TEST_FOLDER):
        print(f"Folder not found: {TEST_FOLDER}"); return

    cursor.execute("DELETE FROM ImageLabels")
    conn.commit()
    
    all_files = [f for f in os.listdir(TEST_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total = len(all_files)
    start_time_total = time.time()

    for index, filename in enumerate(all_files):
        file_path = os.path.join(TEST_FOLDER, filename)
        is_anime_file = any(kw in filename.lower() for kw in ["anime", "waifu", "vn", "kazusa"])
        
        try:
            current_faces = 0
            img = cv2.imread(file_path)
            if img is None: continue

            if is_anime_file:
                anime_faces = get_anime_locations(img)
                for item in anime_faces:
                    x, y, w, h = item['region']
                    cursor.execute("""
                        INSERT INTO ImageLabels (FileName, FilePath, LabelName, Gender, BoxX, BoxY, BoxW, BoxH, FaceData)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)
                    """, (filename, file_path, 'Anime_Character', 'Anime', x, y, w, h))
                    current_faces += 1
            else:
                # InsightFace
                faces = app.get(img)

                for face in faces:
                    box = face.bbox.astype(int)
                    x1, y1, x2, y2 = box
                    w, h = x2 - x1, y2 - y1
                    
                    if w < 20: continue # Skip tiny noise

                    # L2 Normalized Embedding from InsightFace
                    vec = face.normed_embedding
                    vector_str = ",".join(map(str, vec))
                    
                    gender = "Man" if face.gender == 1 else "Woman"
                    
                    cursor.execute("""
                        INSERT INTO ImageLabels (FileName, FilePath, LabelName, Gender, BoxX, BoxY, BoxW, BoxH, FaceData)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (filename, file_path, 'Person', gender, int(x1), int(y1), int(w), int(h), vector_str))
                    current_faces += 1

            conn.commit()
            print(f"[{index+1}/{total}] {filename[:30]:<30} | Detected: {current_faces} | Status: OK")
        except Exception as e:
            print(f"  [!] Failed {filename}: {e}")
            conn.rollback()

    conn.close()
    print("-" * 110)
    print(f"DONE. All images re-encoded with InsightFace in {time.time() - start_time_total:.2f}s")

if __name__ == "__main__":
    process_test_images()
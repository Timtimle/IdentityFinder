import pyodbc
import numpy as np
import cv2
import sys
import io
import os
import shutil
import json
from collections import defaultdict
from scipy.spatial import distance

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

CONN_STR = "Driver={SQL Server};Server=localhost\\SQLEXPRESS;Database=IdentityFinder;Trusted_Connection=yes;"
BASE_DIR = os.getcwd() 
RESULT_IMG_DIR = os.path.join(BASE_DIR, "web_results")

def normalize_vector(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v

def draw_simple_box(img, box, label, color):
    """ Vẽ khung chữ nhật đơn giản nhất - Không banner, không màu mè """
    x, y, w, h = box['x'], box['y'], box['w'], box['h']
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, label, (x, y - 10), font, 0.6, color, 2, cv2.LINE_AA)

def run_hybrid_matching():
    if os.path.exists(RESULT_IMG_DIR):
        shutil.rmtree(RESULT_IMG_DIR)
    os.makedirs(RESULT_IMG_DIR)

    try:
        conn = pyodbc.connect(CONN_STR)
        cursor = conn.cursor()
    except Exception as e:
        print(f"Database Error: {e}")
        return

    cursor.execute("SELECT UserName, FaceData, UserAvatar, Gender FROM Users")
    master_db = {}
    for row in cursor.fetchall():
        name = str(row[0]).capitalize()
        vectors = [normalize_vector(np.fromstring(v, sep=',')) for v in str(row[1]).split(';') if v.strip()]
        if vectors:
            master_db[name] = {"vectors": vectors, "avatar": row[2], "gender": row[3]}
    
    cursor.execute("SELECT FilePath, FaceData, BoxX, BoxY, BoxW, BoxH, FileName, Gender FROM ImageLabels")
    image_groups = defaultdict(list)
    for row in cursor.fetchall():
        path = row[0].replace('¥', '\\')
        image_groups[path].append(row)

    html_items = "" 
    winform_data = [] 

    for real_path, faces in image_groups.items():
        full_img_path = real_path if os.path.isabs(real_path) else os.path.normpath(os.path.join(BASE_DIR, real_path))
        if not os.path.exists(full_img_path): continue
        img = cv2.imread(full_img_path)
        if img is None: continue

        valid_detections = []
        html_tags = []

        for f_row in faces:
            _, f_data, bx, by, bw, bh, filename, test_gender = f_row
            if (bw * bh) < 600: continue
            
            box = {"x": bx, "y": by, "w": bw, "h": bh}

            if test_gender == 'Anime':
                draw_simple_box(img, box, "Anime", (0, 165, 255))
                valid_detections.append({
                    "box": box, "type": "Anime", "name": "Anime Unit",
                    "gender": "Anime", "accuracy": 100.0, "master_avatar": ""
                })
                html_tags.append(f"<span class='tag anime'>Anime</span>")
            
            else:
                if not f_data or not master_db: continue
                curr_enc = normalize_vector(np.fromstring(str(f_data), sep=','))

                scores = []
                for name, info in master_db.items():
                    dists = [distance.cosine(curr_enc, v) for v in info["vectors"]]
                    min_dist = min(dists)
                    votes = sum(1 for d in dists if d < 0.38)
                    scores.append((name, min_dist, info["avatar"], info["gender"], votes))
                
                scores.sort(key=lambda x: x[1])
                b_name, b_dist, b_avatar, b_gender, b_votes = scores[0]
                
                is_match = False
                if b_dist < 0.35:
                    is_match = True
                elif b_dist < 0.40 and b_votes >= 2:
                    is_match = True

                if is_match:
                    acc = max(0, min(99.9, (1 - (b_dist / 0.6)) * 100))
                    draw_simple_box(img, box, f"{b_name} {acc:.1f}%", (0, 255, 0))
                    valid_detections.append({
                        "box": box, "type": "Human", "name": b_name,
                        "gender": b_gender, "accuracy": round(acc, 2), "master_avatar": b_avatar
                    })
                    html_tags.append(f"<span class='tag human'>{b_name} ({acc:.1f}%)</span>")

        if valid_detections:
            res_filename = f"res_{os.path.basename(real_path)}"
            cv2.imwrite(os.path.join(RESULT_IMG_DIR, res_filename), img)
            
            winform_data.append({
                "source_file": os.path.basename(real_path),
                "source_path": real_path,
                "result_web_path": f"web_results/{res_filename}",
                "detections": valid_detections
            })

            tags_html = "".join(html_tags)
            html_items += f"""
            <div class="card">
                <div class="card-header">File: {os.path.basename(real_path)}</div>
                <div class="image-box">
                    <img src="web_results/{res_filename}" alt="Result Image">
                </div>
                <div class="card-info">{tags_html}</div>
            </div>
            """

    with open("results.json", "w", encoding="utf-8") as jf:
        json.dump(winform_data, jf, indent=4, ensure_ascii=False)

    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{ background: #0d1117; color: #c9d1d9; font-family: 'Segoe UI', Tahoma, sans-serif; padding: 20px; }}
            h1 {{ text-align: center; color: #58a6ff; }}
            .container {{ display: flex; flex-direction: column; align-items: center; gap: 20px; }}
            .card {{ background: #161b22; border-radius: 10px; border: 1px solid #30363d; overflow: hidden; max-width: 95%; }}
            .card-header {{ background: #21262d; padding: 10px; font-family: monospace; font-size: 0.9em; }}
            .image-box {{ display: flex; justify-content: center; background: #000; padding: 5px; }}
            /* KHÔNG PHÓNG TO ẢNH - CHỈ HIỆN ĐÚNG SIZE HOẶC NHỎ HƠN */
            img {{ max-width: 100%; height: auto; display: block; }}
            .card-info {{ padding: 15px; display: flex; gap: 8px; flex-wrap: wrap; }}
            .tag {{ padding: 4px 10px; border-radius: 12px; font-size: 0.85em; border: 1px solid; font-weight: bold; }}
            .human {{ color: #3fb950; border-color: rgba(63,185,80,0.4); background: rgba(63,185,80,0.1); }}
            .anime {{ color: #d29922; border-color: rgba(210,153,34,0.4); background: rgba(210,153,34,0.1); }}
        </style>
    </head>
    <body>
        <h1>RECOGNITION REPORT</h1>
        <div class="container">{html_items if html_items else "<h3>No data found.</h3>"}</div>
    </body>
    </html>
    """
    with open("report.html", "w", encoding="utf-8") as f:
        f.write(html_template)

    print(f"[*] DONE: Accuracy & Correct Image Size.")
    conn.close()

if __name__ == "__main__":
    run_hybrid_matching()
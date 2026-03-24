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

# Hỗ trợ tiếng Việt cho Console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

CONN_STR = "Driver={SQL Server};Server=localhost\\SQLEXPRESS;Database=IdentityFinder;Trusted_Connection=yes;"
BASE_DIR = os.getcwd() 
RESULT_IMG_DIR = os.path.join(BASE_DIR, "web_results")

def normalize_vector(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v

def draw_basic_bound(img, box, label, color):
    """ Vẽ khung chữ nhật cơ bản nhất """
    x, y, w, h = box['x'], box['y'], box['w'], box['h']
    
    # 1. Vẽ khung hình chữ nhật bao quanh mặt (Độ dày 2)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

    # 2. Vẽ nhãn tên phía trên khung
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    # Đặt text cách khung 10px về phía trên
    cv2.putText(img, label, (x, y - 10), font, font_scale, color, thickness, cv2.LINE_AA)

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

    # 1. LOAD MASTER DATA
    cursor.execute("SELECT UserName, FaceData, UserAvatar, Gender FROM Users")
    user_rows = cursor.fetchall()
    master_db = {}
    for row in user_rows:
        name = str(row[0]).capitalize()
        vectors = [normalize_vector(np.fromstring(v, sep=',')) for v in str(row[1]).split(';') if v.strip()]
        if vectors:
            master_db[name] = {"vectors": vectors, "avatar": row[2], "gender": row[3]}
    
    # 2. LOAD TARGET IMAGES
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
        match_info_list = []

        for f_row in faces:
            _, f_data, bx, by, bw, bh, filename, test_gender = f_row
            if (bw * bh) < 2500: continue
            
            box = {"x": bx, "y": by, "w": bw, "h": bh}

            if test_gender == 'Anime':
                # Màu vàng hổ phách cho Anime
                draw_basic_bound(img, box, "Anime Unit", (0, 165, 255))
                valid_detections.append({
                    "box": box, "type": "Anime", "name": "Anime Character",
                    "gender": "Anime", "accuracy": 100.0, "master_avatar": ""
                })
                match_info_list.append(f"<span class='tag anime'>Anime</span>")
            
            else:
                if not f_data or not master_db: continue
                curr_enc = normalize_vector(np.fromstring(str(f_data), sep=','))

                scores = []
                for name, info in master_db.items():
                    dists = [distance.cosine(curr_enc, v) for v in info["vectors"]]
                    scores.append((name, min(dists), info["avatar"], info["gender"]))
                
                scores.sort(key=lambda x: x[1])
                best_name, best_dist, best_avatar, master_gender = scores[0]
                
                if best_dist < 0.36:
                    acc = max(0, min(99.9, (1 - (best_dist / 0.6)) * 100))
                    
                    # Màu xanh lá cây cho người (Basic)
                    color = (0, 255, 0) 
                    label_text = f"{best_name} {acc:.1f}%"
                    draw_basic_bound(img, box, label_text, color)

                    valid_detections.append({
                        "box": box, "type": "Human", "name": best_name,
                        "gender": master_gender, "accuracy": round(acc, 2), "master_avatar": best_avatar
                    })
                    match_info_list.append(f"<span class='tag human'>{best_name} ({master_gender})</span>")

        if valid_detections:
            res_filename = f"res_{os.path.basename(real_path)}"
            cv2.imwrite(os.path.join(RESULT_IMG_DIR, res_filename), img)
            
            winform_data.append({
                "source_file": os.path.basename(real_path),
                "source_path": real_path,
                "result_web_path": f"web_results/{res_filename}",
                "detections": valid_detections
            })

            info_html = "".join(match_info_list)
            html_items += f"""<div class="card"><div class="card-header">{os.path.basename(real_path)}</div>
                            <div class="image-container"><img src="web_results/{res_filename}"></div>
                            <div class="card-info">{info_html}</div></div>"""

    with open("results.json", "w", encoding="utf-8") as jf:
        json.dump(winform_data, jf, indent=4, ensure_ascii=False)
    
    print(f"[*] DONE: Basic frames rendered. JSON updated.")
    conn.close()

if __name__ == "__main__":
    run_hybrid_matching()
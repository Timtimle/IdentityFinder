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

    # 1. LOAD MASTER DATA (Lấy thêm Gender gốc từ DB)
    cursor.execute("SELECT UserName, FaceData, UserAvatar, Gender FROM Users")
    user_rows = cursor.fetchall()
    master_db = {}
    for row in user_rows:
        name = str(row[0]).capitalize()
        vectors = [normalize_vector(np.fromstring(v, sep=',')) for v in str(row[1]).split(';') if v.strip()]
        if vectors:
            master_db[name] = {
                "vectors": vectors, 
                "avatar": row[2],
                "gender": row[3] # Lưu gender gốc
            }
    
    print(f"[*] Loaded {len(master_db)} Master Identities.")

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

        valid_detections_in_this_image = []
        match_info_list = []

        for f_row in faces:
            _, f_data, bx, by, bw, bh, filename, test_gender = f_row
            
            if (bw * bh) < 2500: continue

            if test_gender == 'Anime':
                valid_detections_in_this_image.append({
                    "box": {"x": bx, "y": by, "w": bw, "h": bh},
                    "type": "Anime",
                    "name": "Anime Character",
                    "gender": "Anime",
                    "accuracy": 100.0,
                    "master_avatar": ""
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
                    is_certain = True
                    if len(scores) > 1 and (scores[1][1] - best_dist) < 0.05:
                        is_certain = False
                    
                    if is_certain:
                        acc = max(0, min(99.9, (1 - (best_dist / 0.6)) * 100))
                        
                        # Lưu vào JSON (Có thêm Gender)
                        valid_detections_in_this_image.append({
                            "box": {"x": bx, "y": by, "w": bw, "h": bh},
                            "type": "Human",
                            "name": best_name,
                            "gender": master_gender, # Trả về giới tính từ DB
                            "accuracy": round(acc, 2),
                            "master_avatar": best_avatar
                        })
                        # Tag cho HTML
                        gender_color = "#58a6ff" if master_gender == 'Man' else "#ff7b72"
                        match_info_list.append(f"<span class='tag human'>{best_name} ({master_gender}) - {acc:.1f}%</span>")

        if valid_detections_in_this_image:
            res_filename = f"res_{os.path.basename(real_path)}"
            cv2.imwrite(os.path.join(RESULT_IMG_DIR, res_filename), img)
            
            winform_data.append({
                "source_file": os.path.basename(real_path),
                "source_path": real_path,
                "result_web_path": f"web_results/{res_filename}",
                "detections": valid_detections_in_this_image
            })

            info_html = "".join(match_info_list)
            html_items += f"""
            <div class="card">
                <div class="card-header">{os.path.basename(real_path)}</div>
                <div class="image-container"><img src="web_results/{res_filename}"></div>
                <div class="card-info">{info_html}</div>
            </div>
            """

    # Ghi file JSON cho WinForm
    with open("results.json", "w", encoding="utf-8") as jf:
        json.dump(winform_data, jf, indent=4, ensure_ascii=False)
    
    # Ghi file HTML (Style Dark Mode)
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>IdentityFinder Report</title>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; background-color: #0d1117; color: #c9d1d9; padding: 20px; }}
            .container {{ display: flex; flex-direction: column; align-items: center; gap: 20px; }}
            .card {{ background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; width: 90%; max-width: 900px; }}
            .card-header {{ padding: 10px; background: #21262d; border-bottom: 1px solid #30363d; font-family: monospace; }}
            .card-info {{ padding: 10px; display: flex; gap: 10px; }}
            .tag {{ padding: 4px 10px; border-radius: 12px; font-size: 0.8em; border: 1px solid; }}
            .human {{ color: #3fb950; border-color: #3fb950; }}
            .anime {{ color: #d29922; border-color: #d29922; }}
            img {{ width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <h1 style="text-align:center;">AI Recognition Analysis</h1>
        <div class="container">{html_items}</div>
    </body>
    </html>
    """
    with open("report.html", "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"[*] DONE: JSON and HTML updated with Gender info.")
    conn.close()

if __name__ == "__main__":
    run_hybrid_matching()
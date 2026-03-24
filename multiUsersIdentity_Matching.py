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

# Support Vietnamese for console output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# --- CONFIGURATION ---
CONN_STR = "Driver={SQL Server};Server=localhost\\SQLEXPRESS;Database=IdentityFinder;Trusted_Connection=yes;"
BASE_DIR = os.getcwd() 
RESULT_IMG_DIR = os.path.join(BASE_DIR, "web_results")

# Threshold for InsightFace Buffalo-L (Distances > 0.45 will be ignored/hidden)
RECOGNITION_THRESHOLD = 0.45

def normalize_vector(v):
    """ Ensure vector is L2 normalized for consistent Cosine distance calculation """
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v

def draw_simple_box(img, box, label, color):
    """ Draw bounding box and label only for recognized subjects """
    x, y, w, h = int(box['x']), int(box['y']), int(box['w']), int(box['h'])
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, label, (x, y - 12), font, 0.7, color, 2, cv2.LINE_AA)

def run_insight_matching():
    """ 
    Execute matching process. 
    Faces that do not meet the threshold are skipped (no box drawn).
    Full HTML report generation is included.
    """
    if os.path.exists(RESULT_IMG_DIR):
        shutil.rmtree(RESULT_IMG_DIR)
    os.makedirs(RESULT_IMG_DIR)

    try:
        conn = pyodbc.connect(CONN_STR)
        cursor = conn.cursor()
    except Exception as e:
        print(f"Database Error: {e}")
        return

    # ==========================================
    # 1. LOAD MASTER DATA (Known Embeddings)
    # ==========================================
    cursor.execute("SELECT UserName, UserAvatar, Gender FROM Users")
    master_db = {}
    for row in cursor.fetchall():
        name = str(row[0]).capitalize()
        master_db[name] = {"samples": [], "avatar": row[1]}

    cursor.execute("SELECT UserName, FaceData FROM UserEmbeddings")
    for row in cursor.fetchall():
        name = str(row[0]).capitalize()
        if name in master_db and row[1]:
            try:
                vec = normalize_vector(np.fromstring(str(row[1]), sep=','))
                if vec.size == 512:
                    master_db[name]["samples"].append({"vector": vec})
            except: continue

    # ==========================================
    # 2. MATCHING LOGIC (HIDDEN UNKNOWNS)
    # ==========================================
    cursor.execute("SELECT FilePath, FaceData, BoxX, BoxY, BoxW, BoxH, FileName, Gender FROM ImageLabels")
    image_groups = defaultdict(list)
    for row in cursor.fetchall():
        path = row[0].replace('¥', '\\')
        image_groups[path].append(row)

    html_items = "" 
    winform_data = [] 

    print("=" * 110)
    print(f"{'INSIGHTFACE MATCHING (CLEAN MODE + HTML REPORT)':^110}")
    print("=" * 110)

    for real_path, faces in image_groups.items():
        full_img_path = real_path if os.path.isabs(real_path) else os.path.normpath(os.path.join(BASE_DIR, real_path))
        if not os.path.exists(full_img_path): continue
        img = cv2.imread(full_img_path)
        if img is None: continue

        valid_detections = []
        html_tags = []

        for f_row in faces:
            _, f_data, bx, by, bw, bh, filename, test_gender = f_row
            if (bw * bh) < 400: continue 
            
            box = {"x": bx, "y": by, "w": bw, "h": bh}

            if test_gender == 'Anime':
                draw_simple_box(img, box, "Anime", (0, 165, 255))
                valid_detections.append({"box": box, "type": "Anime", "name": "Anime Unit"})
                html_tags.append(f"<span class='tag anime'>Anime</span>")
            else:
                if not f_data or not master_db: continue
                curr_enc = normalize_vector(np.fromstring(str(f_data), sep=','))

                scores = []
                for name, info in master_db.items():
                    if not info["samples"]: continue 
                    dists = [distance.cosine(curr_enc, s["vector"]) for s in info["samples"]]
                    scores.append((name, min(dists), info["avatar"]))
                
                if not scores: continue
                scores.sort(key=lambda x: x[1])
                b_name, b_dist, b_avatar = scores[0]
                
                # --- SELECTIVE DRAWING ---
                if b_dist < RECOGNITION_THRESHOLD:
                    acc = max(0, min(99.9, (1 - (b_dist/1.5)) * 100))
                    draw_simple_box(img, box, f"{b_name} {acc:.1f}%", (0, 255, 0))
                    valid_detections.append({
                        "box": box, "type": "Human", "name": b_name, "accuracy": round(acc, 2), "avatar": b_avatar
                    })
                    html_tags.append(f"<span class='tag human'>{b_name} ({acc:.1f}%)</span>")
                    print(f"    [+] KNOWN: {b_name:<12} | Dist: {b_dist:.3f}")
                else:
                    # Face is unknown -> No drawing, no logging
                    pass

        # Save image and prepare HTML cards
        res_filename = f"res_{os.path.basename(real_path)}"
        cv2.imwrite(os.path.join(RESULT_IMG_DIR, res_filename), img)
        winform_data.append({"source": os.path.basename(real_path), "detections": valid_detections})

        tags_html = "".join(html_tags)
        html_items += f"""
        <div class="card">
            <div class="card-header">File: {os.path.basename(real_path)}</div>
            <div class="image-box"><img src="web_results/{res_filename}"></div>
            <div class="card-info">{tags_html if tags_html else "<span class='tag'>No matches</span>"}</div>
        </div>"""

    # Export JSON results
    with open("results.json", "w", encoding="utf-8") as jf:
        json.dump(winform_data, jf, indent=4, ensure_ascii=False)

    # --- HTML REPORT GENERATION ---
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Identity Finder Report</title>
        <style>
            body {{ background: #0d1117; color: #c9d1d9; font-family: sans-serif; padding: 40px; }}
            h1 {{ text-align: center; color: #58a6ff; }}
            .container {{ display: flex; flex-direction: column; align-items: center; gap: 30px; }}
            .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 12px; overflow: hidden; width: 850px; }}
            .card-header {{ padding: 12px; background: #21262d; font-family: monospace; color: #8b949e; }}
            .image-box {{ display: flex; justify-content: center; background: #000; }}
            img {{ max-width: 100%; height: auto; display: block; }}
            .card-info {{ padding: 15px; display: flex; gap: 10px; flex-wrap: wrap; }}
            .tag {{ padding: 5px 12px; border-radius: 15px; font-size: 0.85em; font-weight: 600; border: 1px solid; }}
            .human {{ color: #3fb950; border-color: rgba(63,185,80,0.4); background: rgba(63,185,80,0.1); }}
            .anime {{ color: #d29922; border-color: rgba(210,153,34,0.4); background: rgba(210,153,34,0.1); }}
        </style>
    </head>
    <body>
        <h1>INSIGHTFACE IDENTIFICATION REPORT</h1>
        <div class="container">{html_items}</div>
    </body>
    </html>
    """
    with open("report.html", "w", encoding="utf-8") as f:
        f.write(html_content)

    conn.close()
    print(f"\n[*] ALL DONE. Report.html generated successfully.")

if __name__ == "__main__":
    run_insight_matching()
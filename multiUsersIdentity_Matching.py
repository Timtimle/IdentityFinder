import pyodbc
import numpy as np
import cv2
import sys
import io
import os
import shutil
from collections import defaultdict
from scipy.spatial import distance

# Hỗ trợ tiếng Việt cho Console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Cấu hình SQL Server
CONN_STR = "Driver={SQL Server};Server=localhost\\SQLEXPRESS;Database=IdentityFinder;Trusted_Connection=yes;"

# Thiết lập thư mục làm việc (Sử dụng GetCurrentDirectory để WinForm dễ nhận diện)
BASE_DIR = os.getcwd() 
RESULT_IMG_DIR = os.path.join(BASE_DIR, "web_results")

def run_hybrid_matching():
    """
    ENGINE NHẬN DIỆN & XUẤT BÁO CÁO WEB:
    - Chạy ngầm hoàn toàn (No imshow).
    - Xuất file report.html Dark Mode.
    - Hiển thị Accuracy thực tế từ FaceNet 512-d.
    """
    
    # 1. Làm sạch thư mục kết quả trước khi chạy
    if os.path.exists(RESULT_IMG_DIR):
        shutil.rmtree(RESULT_IMG_DIR)
    os.makedirs(RESULT_IMG_DIR)

    try:
        conn = pyodbc.connect(CONN_STR)
        cursor = conn.cursor()
    except Exception as e:
        print(f"Database Error: {e}")
        return

    # --- STEP 1: LOAD DỮ LIỆU NGƯỜI QUEN (USERS) ---
    cursor.execute("SELECT UserName, FaceData FROM Users")
    user_rows = cursor.fetchall()
    known_encs, known_names = [], []
    for row in user_rows:
        if row[1]:
            clean_str = str(row[1]).strip().replace(' ', '').replace('\n', '')
            enc = np.fromstring(clean_str, sep=',')
            if enc.size == 512:
                known_encs.append(enc)
                known_names.append(str(row[0]).capitalize())
    
    print(f"[*] Loaded {len(known_encs)} Master Identities.")

    # --- STEP 2: LOAD DỮ LIỆU ẢNH CẦN NHẬN DIỆN (IMAGELABELS) ---
    cursor.execute("SELECT FilePath, FaceData, BoxX, BoxY, BoxW, BoxH, FileName, Gender FROM ImageLabels")
    image_groups = defaultdict(list)
    for row in cursor.fetchall():
        path = row[0].replace('¥', '\\')
        image_groups[path].append(row)

    html_items = "" 

    # --- STEP 3: XỬ LÝ SO KHỚP VÀ VẼ BOX ---
    for real_path, faces in image_groups.items():
        full_img_path = real_path if os.path.isabs(real_path) else os.path.normpath(os.path.join(BASE_DIR, real_path))
        if not os.path.exists(full_img_path): continue
        
        img = cv2.imread(full_img_path)
        if img is None: continue

        any_match = False
        match_info_list = []

        for f_row in faces:
            _, f_data, bx, by, bw, bh, filename, gender = f_row
            label, color = "", (0, 255, 0)

            if gender == 'Anime':
                any_match = True
                label, color = "Anime", (255, 255, 0)
                match_info_list.append(f"<span class='tag anime'>Type: Anime</span>")
            else:
                if f_data is None: continue
                clean_f_data = str(f_data).strip().replace(' ', '').replace('\n', '')
                curr_enc = np.fromstring(clean_f_data, sep=',')
                if curr_enc.size != 512 or not known_encs: continue

                # Tính khoảng cách Cosine (Số càng nhỏ càng giống)
                cosine_dists = [distance.cosine(curr_enc, k_enc) for k_enc in known_encs]
                best_idx = np.argmin(cosine_dists)
                cos_val = cosine_dists[best_idx]
                
                # Ngưỡng nghiêm ngặt 0.38
                if cos_val < 0.38: 
                    any_match = True
                    match_name = known_names[best_idx]
                    
                    # TÍNH TOÁN ACCURACY THỰC TẾ (REAL NUMBERS)
                    acc = (1 - cos_val) * 100 
                    
                    label = f"{match_name} {acc:.1f}%"
                    match_info_list.append(f"<span class='tag human'>Match: {match_name} ({acc:.1f}%)</span>")

            if label:
                # Vẽ khung thanh mảnh (2px)
                cv2.rectangle(img, (bx, by), (bx + bw, by + bh), color, 2)
                # Font chữ nhỏ gọn (0.6)
                cv2.putText(img, label, (bx, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        if any_match:
            # Lưu ảnh kết quả vào folder web_results
            res_filename = f"res_{os.path.basename(real_path)}"
            cv2.imwrite(os.path.join(RESULT_IMG_DIR, res_filename), img)

            # Tạo nội dung HTML Card
            info_html = "".join(match_info_list)
            html_items += f"""
            <div class="card">
                <div class="card-header">File Name: {os.path.basename(real_path)}</div>
                <div class="image-container">
                    <img src="web_results/{res_filename}" alt="Result Image">
                </div>
                <div class="card-info">{info_html}</div>
            </div>
            """

    conn.close()

    # --- STEP 4: XUẤT FILE HTML DARK MODE ---
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>AI Recognition Report</title>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; background-color: #0d1117; color: #c9d1d9; margin: 0; padding: 40px 10px; }}
            h1 {{ text-align: center; color: #58a6ff; font-weight: 300; letter-spacing: 3px; text-transform: uppercase; }}
            .container {{ display: flex; flex-direction: column; align-items: center; gap: 35px; }}
            .card {{ background-color: #161b22; border: 1px solid #30363d; border-radius: 12px; overflow: hidden; width: 95%; max-width: 1000px; box-shadow: 0 8px 24px rgba(0,0,0,0.5); }}
            .card-header {{ background-color: #21262d; padding: 12px 20px; font-size: 0.9em; color: #8b949e; border-bottom: 1px solid #30363d; font-family: monospace; }}
            .image-container {{ width: 100%; overflow-x: auto; background-color: #000; }}
            .card img {{ display: block; max-width: none; height: auto; }}
            .card-info {{ padding: 15px 20px; display: flex; gap: 10px; flex-wrap: wrap; }}
            .tag {{ padding: 5px 15px; border-radius: 15px; font-size: 0.85em; font-weight: 600; border: 1px solid; text-transform: uppercase; }}
            .anime {{ color: #d29922; border-color: rgba(210,153,34,0.4); background: rgba(210,153,34,0.1); }}
            .human {{ color: #3fb950; border-color: rgba(63,185,80,0.4); background: rgba(63,185,80,0.1); }}
            .footer {{ text-align: center; margin-top: 50px; color: #484f58; font-size: 0.8em; }}
        </style>
    </head>
    <body>
        <h1>Recognition System Report</h1>
        <div class="container">{html_items if html_items else "<h3>No matches found in the current session.</h3>"}</div>
        <div class="footer">Generated by IdentityFinder Engine &copy; 2026</div>
    </body>
    </html>
    """

    report_path = os.path.join(BASE_DIR, "report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"REPORT_READY: {report_path}")

if __name__ == "__main__":
    run_hybrid_matching()
import face_recognition
import pyodbc
import numpy as np
import cv2
import sys
import io
import os
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

CONN_STR = "Driver={SQL Server};Server=localhost\\SQLEXPRESS;Database=IdentityFinder;Trusted_Connection=yes;"

def run_hybrid_matching():
    """
    HYBRID MATCHING ENGINE:
    - Humans: Full Identity Recognition (via Euclidean Distance)
    - Anime: Detection Only (Bypassing SQL comparison for Graphical entities)
    """
    try:
        conn = pyodbc.connect(CONN_STR)
        cursor = conn.cursor()
    except Exception as e:
        print(f"Database Connection Error: {e}")
        return

    cursor.execute("SELECT UserName, FaceData FROM Users WHERE Gender = 'Human'")
    user_rows = cursor.fetchall()
    known_encs, known_names = [], []

    for row in user_rows:
        if row[1]:
            enc = np.fromstring(str(row[1]).strip().replace(' ', ''), sep=',')
            if enc.size == 128:
                known_encs.append(enc)
                name = str(row[0]).capitalize()
                known_names.append(name)

    if not known_encs:
        print("SYSTEM LOG: Master Human data is empty. Only Anime detection will work.")

    cursor.execute("SELECT FilePath, FaceData, BoxX, BoxY, BoxW, BoxH, FileName, Gender FROM ImageLabels")
    image_groups = defaultdict(list)
    for row in cursor.fetchall():
        path = row[0].replace('¥', '\\')
        image_groups[path].append(row)

    print("-" * 115)
    print(f"{'FILE NAME':<30} | {'TYPE':<15} | {'RECOGNITION RESULT':<25} | {'STATUS'}")
    print("-" * 115)

    for real_path, faces in image_groups.items():
        if not os.path.exists(real_path): continue
        img = cv2.imread(real_path)
        if img is None: continue

        any_valid_detection = False
        
        for f_row in faces:
            _, f_data, bx, by, bw, bh, filename, gender = f_row
            
            if gender == 'Anime':
                any_valid_detection = True
                display_label = "Anime Face"
                cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (255, 255, 0), 2)
                cv2.putText(img, display_label, (bx, by - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 0), 2)
                print(f"{filename[:30]:<30} | {'Anime':<15} | {'Detected (Graphic)':<25} | OK")
            
            else:
                if f_data is None: continue
                curr_enc = np.fromstring(str(f_data).strip().replace(' ', ''), sep=',')
                if curr_enc.size != 128 or not known_encs: continue

                distances = face_recognition.face_distance(known_encs, curr_enc)
                best_idx = np.argmin(distances)
                dist1 = distances[best_idx]
                
                if dist1 < 0.4:
                    any_valid_detection = True
                    match_name = known_names[best_idx]
                    accuracy = (1 - dist1) * 100
                    
                    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                    cv2.putText(img, f"{match_name} ({accuracy:.1f}%)", (bx, by - 10), 0, 0.7, (0, 255, 0), 2)
                    print(f"{filename[:30]:<30} | {'Human':<15} | {match_name:<25} | MATCH")

        if any_valid_detection:
            h, w = img.shape[:2]
            display_scale = 800 / h if h > 800 else 1.0
            img_disp = cv2.resize(img, (0,0), fx=display_scale, fy=display_scale)
            cv2.imshow("IdentityFinder - Hybrid Mode (Clean View)", img_disp)
            
            if cv2.waitKey(0) == 27: 
                break

    cv2.destroyAllWindows()
    conn.close()
    print("-" * 115)
    print("PROCESS FINISHED: Result stream ended.")

if __name__ == "__main__":
    run_hybrid_matching()
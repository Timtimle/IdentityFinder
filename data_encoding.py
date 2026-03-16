import face_recognition
import pyodbc
import os
import sys
import io
from deepface import DeepFace 

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

CONN_STR = (
    "Driver={SQL Server};"
    "Server=localhost\SQLEXPRESS;"
    "Database=IdentityFinder;"
    "Trusted_Connection=yes;"
)

try:
    conn = pyodbc.connect(CONN_STR)
    cursor = conn.cursor()
    print("System: SQL Connected. Starting Mass Encoding with Gender Detection...")
except Exception as e:
    print(f"Error: Connection failed - {e}")
    exit()
import io
TEST_FOLDER = "test_100_celebA"

if not os.path.exists(TEST_FOLDER):
    print(f"Error: Folder {TEST_FOLDER} not found!")
    exit()

for filename in os.listdir(TEST_FOLDER):
    file_path = os.path.abspath(os.path.join(TEST_FOLDER, filename))
    
    cursor.execute("SELECT LabelID FROM ImageLabels WHERE FileName = ?", (filename,))
    if cursor.fetchone():
        print(f"Skipped: {filename} (Already exists in Database)")
        continue

    image = face_recognition.load_image_file(file_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    if not face_encodings:
        print(f"Skipped: {filename} (No face detected)")
        continue

    try:
        analysis = DeepFace.analyze(img_path=file_path, actions=['gender'], enforce_detection=False)
    except Exception as e:
        print(f"Gender analysis failed for {filename}: {e}")
        analysis = []

    # 3. Store each detected face into SQL
    for i, ((top, right, bottom, left), encoding) in enumerate(zip(face_locations, face_encodings)):
        encoding_str = ",".join(map(str, encoding))
        
        gender = "Unknown"
        if isinstance(analysis, list) and i < len(analysis):
            gender = analysis[i]['dominant_gender'] # Returns 'Man' or 'Woman'
        elif isinstance(analysis, dict): 
            gender = analysis['dominant_gender']

        try:
            cursor.execute("""
                INSERT INTO ImageLabels (FileName, FilePath, LabelName, Gender, BoxX, BoxY, BoxW, BoxH, FaceData)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (filename, file_path, "Pending", gender, left, top, right - left, bottom - top, encoding_str))
            conn.commit()
        except Exception as e:
            print(f"SQL Error at {filename}: {e}")

    print(f"Encoded & Stored: {filename} (Gender: {gender})")

print("--- [FINISH] All 100 images synchronized with Gender data ---")
conn.close()
import face_recognition
import pyodbc
import numpy as np

CONN_STR = "Driver={SQL Server};Server=localhost\SQLEXPRESS;Database=IdentityFinder;Trusted_Connection=yes;"
conn = pyodbc.connect(CONN_STR)
cursor = conn.cursor()

cursor.execute("SELECT UserName, FaceData FROM Users")
user_rows = cursor.fetchall()

known_encodings = []
known_names = []

for row in user_rows:
    known_names.append(row[0])
    encoding_arr = np.fromstring(row[1], sep=',')
    known_encodings.append(encoding_arr)

print(f"System: Loaded {len(known_names)} identities from 'Users' table.")

cursor.execute("SELECT LabelID, FaceData FROM ImageLabels WHERE LabelName = 'Pending' OR LabelName = 'Unknown'")
pending_faces = cursor.fetchall()

print(f"System: Comparing {len(pending_faces)} faces...")

match_count = 0
for row in pending_faces:
    label_id = row[0]
    current_encoding = np.fromstring(row[1], sep=',')
    
    # Compare current face against ALL known encodings
    # tolerance=0.5: Lower is stricter (less false positives)
    results = face_recognition.compare_faces(known_encodings, current_encoding, tolerance=0.5)

    if True in results:
        match_index = results.index(True)
        detected_name = known_names[match_index]

        # UPDATE the LabelName in SQL
        cursor.execute("UPDATE ImageLabels SET LabelName = ? WHERE LabelID = ?", (detected_name, label_id))
        match_count += 1
        print(f"Match Found: {label_id} is identified as '{detected_name}'")

conn.commit()
print(f"--- [FINISH] Identified {match_count} faces in total ---")
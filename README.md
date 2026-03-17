# IdentityFinder

IdentityFinder is a tool for identifying individuals in image collections containing one or more people, while also detecting anime-style faces.
It scans images, detects faces, generates facial embeddings, and matches them against identities stored in a centralized database to automatically label the dataset.

---

## Performance Summary

IdentityFinder leverages a custom YOLOv5s model optimized for Anime faces and RetinaFace for human accuracy.

| Task | Backend | Speed (CPU) | Precision |
| :--- | :--- | :--- | :--- |
| Anime Detect | YOLOv5s | ~50ms/img | 0.95 |
| Human Detect | RetinaFace | ~120ms/img | 0.98 |
| Encoding | dlib (large) | ~15ms/face | 128-d vector |

---

## Installation

Installation is straightforward. You will need Python 3.8+ and a local SQL Server instance.

```bash
pip install torch face_recognition deepface pyodbc numpy opencv-python
```

## Directory Structure

```text
IdentityFinder/
├── anime_detection/       # YOLOv5 core logic & weights
├── known_faces/           # MASTER DATA (Training samples)
│   ├── anime/             # Sub-folder for Anime characters
│   └── person/            # Sub-folder for Real Humans
├── test_100_celebA/       # Target images to be labeled (Inference data)
├── users_encoding.py      # Step 1: Identity Sync (The Brain)
├── data_encoding.py       # Step 2: Mass Extraction (The Worker)
└── multiUsersIdentity.py  # Step 3: Final Matching (The Judge)
```


How to Use

1. Prerequisites

```bash
# Core AI Engines
pip install face_recognition deepface

# Utilities & Database
pip install opencv-python pyodbc numpy
```

2.Database Setup
Create a SQL Server database named IdentityFinder.

Ensure the CONN_STR variable in all .py files matches your local instance (e.g., Server=localhost\SQLEXPRESS).

3.Execution Flow

Step 1: Synchronize Identities (The Brain)
Place high-quality reference images in known_faces/. The script calculates a Master Centroid Vector for each identity using a Refined Box strategy (auto-shrinking the detection box to focus on facial features while ignoring hair interference).

```bash
python users_encoding.py
```

Step 2 : Mass Extraction (The Worker)
Place your unlabeled images in test_100_celebA/. This script uses YOLOv5 (Anime) and RetinaFace (Human) to extract facial landmarks and store them in SQL.

```bash
python data_encoding.py
```

Step 3: Identity Recognition (The Judge)
Run the matcher to compare extracted features against the Master database.

```bash
python multiUsersIdentity.py
```
<img width="841" height="403" alt="image" src="https://github.com/user-attachments/assets/2a6806bc-2b6d-471a-a408-c648a9451a41" />
<img width="429" height="632" alt="image" src="https://github.com/user-attachments/assets/d20a5841-07b5-439e-bbb1-3b83c2973251" />

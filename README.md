# IdentityFinder

IdentityFinder is a tool for identifying individuals in image collections containing one or more people, while also detecting anime-style faces.
It scans images, detects faces, generates facial embeddings, and matches them against identities stored in a centralized database to automatically label the dataset.

---

## Performance Summary

IdentityFinder leverages a custom YOLOv5s model optimized for Anime faces and RetinaFace for human accuracy.

| Task | Backend | Speed (CPU) | Precision |
| :--- | :--- | :--- | :--- |
| Human Analysis | InsightFace (Buffalo-L) | ~150ms/img | 0.99
| Anime Detect | YOLOv5s | ~50ms/img | 0.95 |
| Alignment | 3D Coordinate Warp | ~10ms/face | SOTA |
| Facial Encoding | ArcFace (512-d) | ~30ms/face | Distance-based |

---

## Installation

Installation is straightforward. You will need Python 3.8+ and a local SQL Server instance.

```bash
# SOTA Face Analysis Engine
pip install insightface onnxruntime

# Anime Detection & Utilities
pip install torch opencv-python pyodbc numpy scipy
```

## Directory Structure

```text
IdentityFinder/
├── anime_detection/      # YOLOv5 core logic & custom weights
├── knowns_faces_1/       # MASTER DATA (High-quality samples)
│   └── person/           # Sub-folders named after each identity
├── test_per/             # Target images for inference (Target Data)
├── web_results/          # Generated results with bounding boxes
├── sync_users.py         # Step 1: Identity Enrollment (The Brain)
├── data_encoding.py      # Step 2: Mass Feature Extraction (The Worker)
└── matching.py           # Step 3: Global Identity Matching (The Judge)
```


How to Use

Prerequisites

```bash
# Core AI Engines
pip install face_recognition deepface

# Utilities & Database
pip install opencv-python pyodbc numpy
```

Database Setup

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
<img width="841" height="807" alt="image" src="https://github.com/user-attachments/assets/8726a5ef-0bcc-4cbb-9404-ae6f53c7cb53" />
<img width="706" height="533" alt="image" src="https://github.com/user-attachments/assets/c502ac7d-e5e4-41aa-a2c1-2b72c779b9da" />


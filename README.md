# IdentityFinder

The simplest, fastest tool for auto-labeling mixed datasets of Anime characters and Humans.

This repository provides a pragmatic, hybrid approach to image labeling. Instead of manual clicking, **IdentityFinder** uses AI to scan, encode, and identify faces against a centralized SQL database. The code is designed to be plain, readable, and highly effective for developers handling large image collections.

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

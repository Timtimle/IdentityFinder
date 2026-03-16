# Automated Image Labeling System (AILS)

The simplest, fastest tool for auto-labeling mixed datasets of Anime characters and Humans.

This repository provides a pragmatic, hybrid approach to image labeling. Instead of manual clicking, it uses AI to scan, encode, and identify faces against a centralized SQL database. The code is designed to be plain, readable, and highly effective for developers handling large image collections.

---

## Performance Summary

The system leverages a custom YOLOv5s model optimized for Anime faces and RetinaFace for human accuracy.

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
Key Dependencies:

PyTorch: The engine behind YOLOv5.

SQL Server: Stores the "Master Vectors" (your identity database).

DeepFace: Handles gender detection and human face extraction.

Quick Start
If you have your sample images ready in the known_faces/ folder, follow these three steps to process your data:

1. Synchronize Identities
The AI scans your known folders, calculates the mean vector for each person, and saves the data to SQL. This establishes the "Master" reference for each identity.

Bash
python sync_master.py
2. Mass Processing
Drop your unlabeled images into the target folder and let the script scan and extract facial features. All extracted data will be stored in the ImageLabels table.

Bash
python mass_encoding.py
3. Review Results
Run the matcher to compare the extracted features against the Master database. The system will display the identified names along with an accuracy score. Press ESC to stop, or any key to see the next result.

Bash
python match_users.py
Technical Concept: Master Centroid Strategy
The system does not perform recognition based on single-image references. It uses a Master Centroid strategy to ensure stability:

It collects all feature vectors from a specific folder (e.g., Touma Kazusa).

It calculates the mean vector (Centroid) for that identity to reduce noise.

For unlabeled images, it calculates the Euclidean Distance against these Centroids.

If Distance < 0.4, the identity is verified.

Python
# Core logic for accuracy calculation:
accuracy = (1 - distance) * 100
if distance < 0.4:
    final_name = "Identified"
Directory Structure
Plaintext
.
├── anime_detection/   # YOLOv5 architecture & weights
├── known_faces/       # Place training samples here (Anime/ or Person/)
├── test_images/       # Place your unlabeled target images here
├── sync_master.py     # Step 1: Identity creator
├── mass_encoding.py   # Step 2: Feature extractor
└── match_users.py     # Step 3: Verification & Preview
Notes on Accuracy
If you find too many "Unknown" results, consider lowering the num_jitters in sync_master.py or adjusting the 0.4 distance threshold in match_users.py to fit your specific dataset quality. Higher quality images in the known_faces/ folder will result in more stable Master Vectors.

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

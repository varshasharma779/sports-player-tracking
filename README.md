#  Player Tracking using YOLOv11 + DeepSORT

This project performs multi-player tracking in soccer videos using **YOLOv11** for detection and **DeepSORT** for consistent ID assignment. It includes Non-Maximum Suppression (NMS) to remove overlapping detections and exports tracking results as annotated videos and CSV files. Ideal for sports analytics and re-identification tasks.

---

##  Features

- YOLOv11 for accurate player detection
- DeepSORT for tracking with consistent IDs
- Removes overlapping boxes using OpenCV NMS
- Outputs: annotated videos + player tracking CSVs
- Supports batch video processing

---
##  Setup in VS Code

###  Prerequisites

- Python 3.8 or above
- Git (optional but useful)
- VS Code installed with Python extension

### 1. Clone the Repository

```bash
git clone https://github.com/varshasharma779/sports-player-tracking/tree/main

### 2. Create Virtual Environment

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not present, install manually:

```bash
pip install ultralytics
pip install deep_sort_realtime
pip install opencv-python
pip install numpy
```

---

##  Running the Project in VS Code

1. Open the folder in VS Code.
2. Ensure the virtual environment is selected (Ctrl+Shift+P → Python: Select Interpreter).
3. Place your YOLOv11 weight file as `best.pt` in the project directory.
4. Add videos (e.g., `tacticam.mp4`) to the root folder.
5. Run the script:

```bash
python tracking_script.py
```

6. Press `q` to exit the live preview.

---

##  Output

- `output_<video>.avi` → video with bounding boxes and player IDs
- `<video>_tracking.csv` → coordinates and track IDs

CSV Format:

| frame | track_id | x1 | y1 | x2 | y2 | cx | cy |
|-------|----------|----|----|----|----|----|----|

---

##  Techniques Used

- **YOLOv11**: real-time object detection.
- **DeepSORT**: re-ID + Kalman Filter-based tracker.
- **NMS**: used to remove overlapping bounding boxes with `cv2.dnn.NMSBoxes`.

---

##  Challenges Encountered

-  Norfair tracker caused frequent ID fluctuation.
-  Without overlap removal, same player got multiple IDs.
-  IDs still fluctuate during occlusion (e.g., player hidden behind another).

---

##  Future Improvements

- Integrate stronger Re-ID models for better identity preservation.
- Use transformer-based tracking methods for occlusion handling.
- Tune DeepSORT parameters (`max_age`, `nms_max_overlap`) for longer memory.
- Explore multi-view camera setups for robust identification.

---

## requirements.txt

```txt
ultralytics
deep_sort_realtime
opencv-python
numpy
```

---

##  Repository Info

-  Last updated: 2025-07-11
-  Repo name: `player-tracking-yolov11-deepsort`
-  Ideal for: Sports analytics, soccer video analysis, ID-based player tracking

---

##  Author

Made by Varsha Sharma  
Robotics & AI, YMCA Faridabad  
Contact via GitHub or email (optional)



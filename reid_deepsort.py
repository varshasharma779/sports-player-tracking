
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import csv
import os


import numpy as np

def remove_overlapping_boxes(detections, iou_threshold=0.6):
    """Remove overlapping bounding boxes using IoU and keep the highest confidence one."""
    if not detections:
        return []

    boxes = np.array([d[0] for d in detections])
    scores = np.array([d[1] for d in detections])

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    x2 = x1 + w
    y2 = y1 + h

    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes.tolist(),
        scores=scores.tolist(),
        score_threshold=0.5,
        nms_threshold=iou_threshold
    )

    if len(indices) == 0:
        return []

    indices = indices.flatten()
    return [detections[i] for i in indices]



# --- Load YOLOv11 detection model ---
model = YOLO("best.pt")  # Ensure 'best.pt' detects 'player' class

# --- DeepSORT tracker setup with appearance embedding ---
tracker = DeepSort(
    max_age=30,
    n_init=3,
    nn_budget=100,
    embedder="mobilenet",
    embedder_gpu=True,
    half=True,
    bgr=True,
    nms_max_overlap=0.7
)

# --- Attempt to locate player class ID from YOLO model ---
CLASS_NAMES = model.names
PLAYER_CLASS_ID = None
for cid, cname in CLASS_NAMES.items():
    if "player" in cname.lower() or cname.lower() == "person":
        PLAYER_CLASS_ID = cid
        break
if PLAYER_CLASS_ID is None:
    raise ValueError("No 'player' class found in YOLO model.")

# --- List of videos ---
videos = ["15sec_input_720p.mp4", "broadcast.mp4", "tacticam.mp4"]

for video_file in videos:
    cap = cv2.VideoCapture(video_file)

    # Read one frame to get resolution
    ret, frame = cap.read()
    if not ret:
        print(f"Cannot open {video_file}")
        continue

    height, width = frame.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Output video and CSV
    base = os.path.splitext(video_file)[0]
    out = cv2.VideoWriter(f"output_{base}.avi", cv2.VideoWriter_fourcc(*"XVID"), 30, (width, height))
    csv_file = open(f"{base}_tracking.csv", mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "track_id", "x1", "y1", "x2", "y2", "cx", "cy"])

    print(f"\nProcessing {video_file}...")

    frame_num = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        # Run YOLOv11 detection
        results = model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])

            if cls_id != PLAYER_CLASS_ID or conf < 0.5:
                continue  # Filter only confident player detections

            w, h = x2 - x1, y2 - y1
            detections.append(([x1, y1, w, h], conf, 'player'))

        # DeepSORT tracking with embeddings
        # tracks = tracker.update_tracks(detections, frame=frame)
            filtered_detections = remove_overlapping_boxes(detections)
            tracks = tracker.update_tracks(filtered_detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, r, b = map(int, track.to_ltrb())
            cx, cy = (l + r) // 2, (t + b) // 2

            # Draw and export
            cv2.rectangle(frame, (l, t), (r, b), (255, 0, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            csv_writer.writerow([frame_num, track_id, l, t, r, b, cx, cy])

        out.write(frame)
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    csv_file.close()
    print(f"Done: output_{base}.avi + {base}_tracking.csv")

cv2.destroyAllWindows()


# from ultralytics import YOLO
# from deep_sort_realtime.deepsort_tracker import DeepSort
# import cv2
# import csv
# import os
# import numpy as np

# # === Remove overlapping boxes using OpenCV NMS ===
# def remove_overlapping_boxes(detections, iou_threshold=0.6):
#     if not detections:
#         return []

#     boxes = np.array([d[0] for d in detections])
#     scores = np.array([d[1] for d in detections])

#     indices = cv2.dnn.NMSBoxes(
#         bboxes=boxes.tolist(),
#         scores=scores.tolist(),
#         score_threshold=0.5,
#         nms_threshold=iou_threshold
#     )

#     if len(indices) == 0:
#         return []

#     indices = indices.flatten()
#     return [detections[i] for i in indices]

# # === Load YOLOv11 ===
# model = YOLO("best.pt")

# # === Initialize DeepSORT ===
# tracker = DeepSort(
#     max_age=30,
#     n_init=3,
#     nn_budget=100,
#     embedder="mobilenet",
#     embedder_gpu=True,
#     half=True,
#     bgr=True,
#     nms_max_overlap=0.7
# )

# # === Get 'person' class ID ===
# CLASS_NAMES = model.names
# PLAYER_CLASS_ID = None
# for cid, cname in CLASS_NAMES.items():
#     if "person" in cname.lower() or "player" in cname.lower():
#         PLAYER_CLASS_ID = cid
#         break
# if PLAYER_CLASS_ID is None:
#     raise ValueError("No 'player' class found in YOLO model.")

# # === List of input videos ===
# videos = ["15sec_input_720p.mp4", "broadcast.mp4", "tacticam.mp4"]

# for video_file in videos:
#     cap = cv2.VideoCapture(video_file)
#     ret, frame = cap.read()
#     if not ret:
#         print(f"⚠️ Cannot read {video_file}")
#         continue

#     height, width = frame.shape[:2]
#     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

#     base = os.path.splitext(video_file)[0]
#     out = cv2.VideoWriter(f"output_{base}.avi", cv2.VideoWriter_fourcc(*"XVID"), 30, (width, height))
#     csv_file = open(f"{base}_tracking.csv", mode='w', newline='')
#     csv_writer = csv.writer(csv_file)
#     csv_writer.writerow(["frame", "track_id", "x1", "y1", "x2", "y2", "cx", "cy"])

#     print(f"▶️ Processing {video_file}...")
#     frame_num = 0

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame_num += 1

#         # --- YOLO detection ---
#         results = model(frame, verbose=False)[0]
#         raw_detections = []
#         for box in results.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             conf = float(box.conf[0])
#             cls_id = int(box.cls[0])

#             if cls_id != PLAYER_CLASS_ID or conf < 0.5:
#                 continue
#             w, h = x2 - x1, y2 - y1
#             raw_detections.append(([x1, y1, w, h], conf, 'player'))

#         # --- Apply NMS ---
#         detections = remove_overlapping_boxes(raw_detections)

#         # --- Update DeepSORT ---
#         tracks = tracker.update_tracks(detections, frame=frame)

#         for track in tracks:
#             if not track.is_confirmed():
#                 continue

#             track_id = track.track_id
#             l, t, r, b = map(int, track.to_ltrb())
#             cx, cy = (l + r) // 2, (t + b) // 2

#             # --- Draw ---
#             cv2.rectangle(frame, (l, t), (r, b), (255, 0, 0), 2)
#             cv2.putText(frame, f"ID {track_id}", (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

#             # --- Save to CSV ---
#             csv_writer.writerow([frame_num, track_id, l, t, r, b, cx, cy])

#         out.write(frame)
#         cv2.imshow("Tracking", frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#     cap.release()
#     out.release()
#     csv_file.close()
#     print(f"✅ Done: output_{base}.avi + {base}_tracking.csv")

# cv2.destroyAllWindows()


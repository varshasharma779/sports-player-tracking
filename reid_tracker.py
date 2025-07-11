
from ultralytics import YOLO
import cv2
from norfair import Detection, Tracker
import numpy as np
import os

# Load model
model = YOLO("best.pt")  # Make sure 'best.pt' is in the same folder

# Norfair tracker setup
tracker = Tracker(distance_function="euclidean", distance_threshold=30)

# Define this function BEFORE you use it
def to_norfair_detections(boxes):
    return [Detection(points=np.array([[(x1 + x2)/2, (y1 + y2)/2]])) for x1, y1, x2, y2 in boxes]

# List of input videos
video_list = ["15sec_input_720p.mp4", "broadcast.mp4", "tacticam.mp4"]

for video_name in video_list:
    cap = cv2.VideoCapture(video_name)
    
    # Read one frame to get resolution
    ret, frame = cap.read()
    if not ret:
        print(f"Could not read from {video_name}. Skipping...")
        continue

    height, width = frame.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset back to first frame

    # Define video writer with same resolution as input
    out = cv2.VideoWriter(f"output_{video_name}", cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))

    print(f"\nProcessing {video_name}...\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()

            detections = to_norfair_detections(boxes)
            tracked_objects = tracker.update(detections)

            for obj, box in zip(tracked_objects, boxes):
                # Draw bounding box
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # Draw center and ID
                x, y = map(int, obj.estimate[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(frame, f'ID {obj.id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out.write(frame)
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()

cv2.destroyAllWindows()

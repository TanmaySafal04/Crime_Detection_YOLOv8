import cv2
from ultralytics import YOLO
import numpy as np
# Path to the output video file
model = YOLO('datasets\Crime_best.pt')
output_path = 'anotated_video/crime_video.mp4'

video_path = 'Clerk bombarded by 4 gunmen during robbery at SW Houston gas station.mp4'
# Open the input video file
cap = cv2.VideoCapture(video_path)

# Get the video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
colors = np.random.uniform(0, 255, size=(len(model.names), 3))
# Perform object detection and write the processed frames
results = model.predict(video_path, stream=True)

for result in results:
    frame = result.orig_img  # Original frame

    # Iterate through the detections
    for box in result.boxes:
        # Unpack the bounding box coordinates, confidence score, and class label
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0]
        cls = int(box.cls[0])

        # Get the label name
        label = model.names[cls]
        color = colors[cls]

        # Draw the bounding box and label on the frame
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)            
        cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Write the processed frame to the output video
    out.write(frame)

# Release the video capture and writer objects
cap.release()
out.release()

print(f'Processed video saved to {output_path}')

import cv2
from ultralytics import YOLO
import numpy as np

# Load your custom-trained YOLO model
model = YOLO('best.pt')

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")
colors = np.random.uniform(0, 255, size=(len(model.names), 3))

# Perform object detection and display the results in real-time
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Perform prediction
    results = model.predict(frame)

    # Process the results
    for result in results:
        for box in result.boxes:
            # Unpack the bounding box coordinates, confidence score, and class label
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])

            # Get the label name
            label = model.names[cls]
            color = colors[cls]

            # Draw the bounding box and label on the frame
            #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)            
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the frame with detections
    cv2.imshow('Object Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the display window
cap.release()
cv2.destroyAllWindows()

print("Webcam closed.")

import cv2
import cvzone
import numpy as np
from ultralytics import YOLO

# Load the YOLO model for weapon detection
model = YOLO('yolo_weights/yolov8l.pt') 

# Start webcam capture
cap = cv2.VideoCapture(0)  # Webcam is used, so no need for path here

knife_class_id = 43  # Change this if necessary for your specific model

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    # Use YOLO to detect objects in the frame
    results = model(frame)

    # Access the detections
    detections = results[0].boxes  # Access the boxes (bounding boxes)

    for detection in detections:
        class_id = int(detection.cls)  # Get class ID from the result

        if class_id == knife_class_id:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = detection.xyxy[0].tolist()  # Convert tensor to list and unpack
            
            # Convert coordinates to integers
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Draw rectangle around the knife and label
            cvzone.cornerRect(frame, (x1, y1, x2 - x1, y2 - y1), l=9, rt=3)
            cvzone.putTextRect(frame, "Knife", (x1, y1 - 10), scale=1, thickness=2)

    # Display the frame with knife detection
    cv2.imshow("Knife Detection", frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()

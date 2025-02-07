from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
from sort import *

cap = cv2.VideoCapture("data/Videos/cars.mp4") 

model = YOLO('yolo_weights/yolov8l.pt') 

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Load and resize the mask to match the frame size
mask = cv2.imread("data/Images/mask_car.png", 0)  # Ensure it's grayscale
total_counts = []
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
limits = [400 , 297, 673, 297]

while True:
    success, img = cap.read()
    if not success:
        print("Video has ended or failed to read.")
        break

    # Resize mask to match the size of the frame
    mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))

    # Ensure `mask` and `img` are compatible
    imgRegion = cv2.bitwise_and(img, img, mask=mask_resized)

    imgGraphics = cv2.imread("data/Images/graphics_car.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (0, 0))

    # Initialize detections array
    detections = np.empty((0, 5))

    results = model(imgRegion, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass in ["car", "bus", "truck", "motorbike"] and conf > 0.3:
                # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultTracker = tracker.update(detections)
    cv2.line(img, (limits[0], limits[1]),(limits[2], limits[3]), (0,0,255), 5)

    for result in resultTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f'{id}', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)

        cx, cy = x1+w //2, y1+h//2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED  )

        if limits[0] <cx < limits[2] and limits[1] -30 <cy < limits[1] +30:
            if total_counts.count(id) == 0:
              total_counts.append(id )
              cv2.line(img, (limits[0], limits[1]),(limits[2], limits[3]), (0, 255, 0), 5)

            
    # cvzone.putTextRect(img, f'Count: {len(total_counts)}', (50, 50))
    cv2.putText(img, str(len(total_counts)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255),8)
    # Display the images
    cv2.imshow("Image", img)
    # cv2.imshow("ImageRegion", imgRegion)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

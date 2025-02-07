import cv2
from cvzone.FaceDetectionModule import FaceDetector
from scipy.spatial import distance
from ultralytics import YOLO

# Load YOLO model for person detection
def load_yolo_model():
    return YOLO('yolo_weights/yolov8n.pt')  # Updated to the correct model path

# Load face detection model
def load_face_detector():
    return FaceDetector()

# Get the face bounding box of the target person from a reference image
def get_face_bbox(image_path, face_detector):
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Unable to load the reference image.")
        return None

    _, faces = face_detector.findFaces(img)
    if faces:
        return faces[0]['bbox']
    print("Error: No face detected in the reference image.")
    return None

# Check if two bounding boxes represent the same person
def is_same_person(face_bbox, ref_bbox, threshold=40):
    ref_center = [(ref_bbox[0] + ref_bbox[2]) / 2, (ref_bbox[1] + ref_bbox[3]) / 2]
    face_center = [(face_bbox[0] + face_bbox[2]) / 2, (face_bbox[1] + face_bbox[3]) / 2]
    return distance.euclidean(ref_center, face_center) < threshold

# Detect the target person from webcam
def detect_person_from_webcam(yolo_model, face_detector, ref_face_bbox):
    cap = cv2.VideoCapture(0)  # Open webcam
    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return

    print("Starting webcam detection. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from webcam.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect people using YOLO
        results = yolo_model(frame_rgb)
        persons = []
        for result in results[0].boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = result
            if int(cls) == 0:  # Class 0 is 'person' in YOLO
                persons.append((x1, y1, x2, y2, conf))

        # Detect faces in the frame
        _, faces = face_detector.findFaces(frame)

        detected = False
        if faces:
            for face in faces:
                face_bbox = face['bbox']
                if is_same_person(face_bbox, ref_face_bbox):
                    detected = True
                    cv2.rectangle(frame, (int(face_bbox[0]), int(face_bbox[1])), 
                                  (int(face_bbox[0] + face_bbox[2]), int(face_bbox[1] + face_bbox[3])),
                                  (0, 255, 0), 2)
                    cv2.putText(frame, "Target Lock!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (0, 0, 255), 2)
        
        if not detected:
            cv2.putText(frame, "Target Person not Detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow("Webcam Detection", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main
if __name__ == "__main__":
    # Load models
    yolo_model = load_yolo_model()
    face_detector = load_face_detector()

    # Path to reference image
    ref_image_path = "data/Images/4.jpg"  # Updated to the correct reference image path

    # Get the reference face bounding box
    ref_face_bbox = get_face_bbox(ref_image_path, face_detector)

    if ref_face_bbox:
        # Detect the specific person from webcam
        detect_person_from_webcam(yolo_model, face_detector, ref_face_bbox)

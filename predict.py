import cv2
import mediapipe as mp
import json

# Load model (ensure models exist)
recognizer = None
try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("models/lbph_model.xml")
except Exception:
    recognizer = None

# Load label map
label_map = {}
try:
    with open("models/label_map.json", "r") as f:
        label_map = json.load(f)
except Exception:
    label_map = {}

mp_face = mp.solutions.face_mesh

cap = cv2.VideoCapture(0)

# Allow up to 5 faces (so at least two people are recognized when present)
with mp_face.FaceMesh(
        max_num_faces=5,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
) as fm:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = fm.process(rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                xs = [int(lm.x * w) for lm in face_landmarks.landmark]
                ys = [int(lm.y * h) for lm in face_landmarks.landmark]
                x_min, x_max = max(min(xs), 0), min(max(xs), w - 1)
                y_min, y_max = max(min(ys), 0), min(max(ys), h - 1)

                # Draw rectangle around face
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Prepare face for prediction; skip tiny or empty crops
                face_crop = frame[y_min:y_max, x_min:x_max]
                if face_crop is None or face_crop.size == 0:
                    continue
                if face_crop.shape[0] < 20 or face_crop.shape[1] < 20:
                    # too small to recognize reliably
                    cv2.putText(frame, "Too small", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    continue

                try:
                    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                except Exception:
                    continue

                text = "Unknown"
                if recognizer is not None:
                    try:
                        # LBPH predict can throw if image is incompatible; guard it
                        label_id, confidence = recognizer.predict(gray)
                        name = label_map.get(str(label_id), None)
                        if name:
                            text = f"{name} ({int(confidence)})"
                        else:
                            text = f"ID:{label_id} ({int(confidence)})"
                    except Exception:
                        text = "Unknown"

                cv2.putText(frame, text, (x_min, max(y_min - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        else:
            # Optionally show a message when no faces are found
            pass

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

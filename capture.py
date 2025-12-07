import cv2
import mediapipe as mp
import os

def capture_person(person_name):
    """Capture face images for a single person."""
    save_dir = f"dataset/{person_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    mp_face = mp.solutions.face_mesh
    cap = cv2.VideoCapture(0)
    count = 0
    
    print(f"\n--- Capturing faces for: {person_name} ---")
    print("Face yourself toward the camera.")
    print("Press SPACE to capture frames, Q to finish.")
    
    with mp_face.FaceMesh(
            max_num_faces=1,
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
            
            # Display instruction text
            cv2.putText(frame, f"Capturing: {person_name}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Images saved: {count}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "SPACE=capture  Q=done", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    xs = [int(lm.x * w) for lm in face_landmarks.landmark]
                    ys = [int(lm.y * h) for lm in face_landmarks.landmark]
                    x_min, x_max = max(min(xs), 0), min(max(xs), w - 1)
                    y_min, y_max = max(min(ys), 0), min(max(ys), h - 1)
                    
                    # Draw green rectangle
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max),
                                  (0, 255, 0), 2)
            
            cv2.imshow("Face Capture", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # SPACE to capture
            if key == ord(' '):
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        xs = [int(lm.x * w) for lm in face_landmarks.landmark]
                        ys = [int(lm.y * h) for lm in face_landmarks.landmark]
                        x_min, x_max = max(min(xs), 0), min(max(xs), w - 1)
                        y_min, y_max = max(min(ys), 0), min(max(ys), h - 1)
                        
                        face_crop = frame[y_min:y_max, x_min:x_max]
                        if face_crop.size > 0:
                            cv2.imwrite(f"{save_dir}/{count}.jpg", face_crop)
                            count += 1
                            print(f"  Saved image {count}")
            
            # Q to quit
            elif key == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n✓ Saved {count} images for {person_name}\n")
    return count


# Main workflow: Capture 2 people
print("=" * 50)
print("Face Recognition: 2-Person Setup")
print("=" * 50)

people = []
for i in range(2):
    while True:
        name = input(f"\nEnter name for Person {i+1}: ").strip()
        if name:
            break
        print("Name cannot be empty. Please try again.")
    
    count = capture_person(name)
    if count > 0:
        people.append(name)
    else:
        print(f"Warning: No images captured for {name}")

print("\n" + "=" * 50)
print(f"Capture complete! People registered: {people}")
print("Next step: Run 'python train.py' to train the model")
print("Then run 'python predict.py' to recognize faces")
print("=" * 50)

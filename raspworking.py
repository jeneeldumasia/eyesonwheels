import cv2
import numpy as np
import pygame
import time
from picamera2 import Picamera2

# === Alarm Setup ===
pygame.init()
pygame.mixer.init()
alarm = pygame.mixer.Sound('alarm.wav')
alarm_playing = False

# === Thresholds ===
EYE_MISSING_FRAMES = 15
eye_missing_count = 0

# === Load Haar Cascades ===
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

# === Start Camera ===
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
picam2.start()
time.sleep(1)

while True:
    frame = picam2.capture_array()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # === Detect Faces ===
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        cv2.putText(frame, "No face detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        eye_missing_count += 1
    else:
        for (x, y, w, h) in faces:
            face_roi_gray = gray[y:y+h, x:x+w]
            face_roi_color = frame[y:y+h, x:x+w]

            # === Draw Face Box ===
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # === Detect Eyes Inside Face ===
            eyes = eye_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=3)
            
            # Filter eyes based on position (roughly upper half of face, not overlapping center line)
            valid_eyes = [(ex, ey, ew, eh) for (ex, ey, ew, eh) in eyes if ey + eh < h // 2]


            if len(valid_eyes) >= 1:
                eye_missing_count = 0
                cv2.putText(frame, "Eyes open!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                for (ex, ey, ew, eh) in valid_eyes:
                    cv2.rectangle(face_roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

                if alarm_playing:
                    alarm.stop()
                    alarm_playing = False
            else:
                eye_missing_count += 1
                cv2.putText(frame, "Eyes not detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # === Trigger Alarm if Eyes Missing Too Long ===
    if eye_missing_count >= EYE_MISSING_FRAMES and not alarm_playing:
        alarm.play()
        alarm_playing = True

    cv2.imshow("EyesOnWheels (Haar)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
model = tf.keras.models.load_model("model.h5", compile=False)
with open("class_names.txt") as f:
    classes = [line.strip() for line in f]

print("Model Output Shape:", model.output_shape)
print("Number of Classes:", len(classes))

base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)
landmarker = vision.HandLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    h, w, _ = frame.shape
    predicted_letter = ""

    # Convert frame to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )
    result = landmarker.detect(mp_image)
    if result.hand_landmarks:
        hand_landmarks = result.hand_landmarks[0]
        xs, ys = [], []
        for lm in hand_landmarks:
            xs.append(int(lm.x * w))
            ys.append(int(lm.y * h))
        pad = 25
        x1 = max(0, min(xs) - pad)
        y1 = max(0, min(ys) - pad)
        x2 = min(w, max(xs) + pad)
        y2 = min(h, max(ys) + pad)
        hand_crop = frame[y1:y2, x1:x2]
        if hand_crop.size != 0:
            hand_crop = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)
            hand_crop = cv2.resize(hand_crop, (48, 48))
            hand_crop = hand_crop.astype("float32") / 255.0
            hand_crop = np.expand_dims(hand_crop, axis=-1)
            hand_crop = np.expand_dims(hand_crop, axis=0)
            pred = model.predict(hand_crop, verbose=0)[0]
            class_id = int(np.argmax(pred))
            confidence = float(np.max(pred))
            if class_id < len(classes):
                predicted_letter = classes[class_id]
            else:
                predicted_letter = ""
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )
    cv2.putText(
        frame,
        predicted_letter,
        (50, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 255, 0),
        4
    )

    cv2.imshow("ASL Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
landmarker.close()
cv2.destroyAllWindows()

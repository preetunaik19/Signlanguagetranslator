import os
import cv2
import mediapipe as mp
import numpy as np

# Directory to save dataset
DATA_DIR = "./sign_data"
if not os.path.exists(DATA_DIR):
  os.makedirs(DATA_DIR)

# Define the signs you want to recognize
CLASSES = ["A", "B", "C", "HELLO"]
dataset_size = 150  # Number of data samples per sign

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7
)

cap = cv2.VideoCapture(0)

for cls in CLASSES:
  class_dir = os.path.join(DATA_DIR, cls)
  if not os.path.exists(class_dir):
    os.makedirs(class_dir)

  print(f"\n--- Collecting data for sign: '{cls}' ---")
  print("Press 's' when you are ready to start recording data.")

  while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2.putText(
        frame,
        f"Ready for: {cls}? Press 's' to start",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    cv2.imshow("Data Collection", frame)
    if cv2.waitKey(25) & 0xFF == ord("s"):
      break

  counter = 0
  while counter < dataset_size:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
        )

        landmark_list = []
        for lm in hand_landmarks.landmark:
          landmark_list.extend([lm.x, lm.y, lm.z])

        np.save(
            os.path.join(class_dir, f"{counter}.npy"),
            np.array(landmark_list),
        )
        counter += 1

    cv2.putText(
        frame,
        f"Collected: {counter}/{dataset_size} for {cls}",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
    )
    cv2.imshow("Data Collection", frame)
    if cv2.waitKey(25) & 0xFF == ord("q"):
      break

cap.release()
cv2.destroyAllWindows()
print("Data collection finished!")
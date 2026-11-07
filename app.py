import cv2
import joblib
import mediapipe as mp
import numpy as np

# Load the trained model
model_dict = joblib.load("sign_model.p")
model = model_dict  # If saved directly as model object

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7
)

cap = cv2.VideoCapture(0)
print("Starting real-time translator... Press 'q' to exit.")

while cap.isOpened():
  ret, frame = cap.read()
  if not ret:
    break

  frame = cv2.flip(frame, 1)
  H, W, _ = frame.shape
  rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  results = hands.process(rgb_frame)
  if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
      mp_drawing.draw_landmarks(
          frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
      )

      # Extract coordinates for prediction
      landmark_list = []
      for lm in hand_landmarks.landmark:
        landmark_list.extend([lm.x, lm.y, lm.z])

      # Predict the sign
      prediction = model.predict([np.array(landmark_list)])
      predicted_sign = prediction[0]

      # Display the prediction on the screen
      cv2.putText(
          frame,
          f"Sign: {predicted_sign}",
          (50, 100),
          cv2.FONT_HERSHEY_SIMPLEX,
          1.3,
          (0, 255, 0),
          3,
          cv2.LINE_AA,
      )

  cv2.imshow("Sign Language Translator", frame)
  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

cap.release()
cv2.destroyAllWindows()
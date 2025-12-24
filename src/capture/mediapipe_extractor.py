import cv2
import mediapipe as mp
import numpy as np

# Suppress MediaPipe protobuf warnings internally
import warnings
warnings.filterwarnings("ignore")

mp_hands = mp.solutions.hands

class LandmarkExtractor:
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1  # Ensure model uses standard complexity (0=Lite, 1=Full)
        )

    def normalize_hand(self, landmarks):
        coords = [[lm.x, lm.y] for lm in landmarks]
        base_x, base_y = coords[0][0], coords[0][1]
        
        relative_coords = []
        for x, y in coords:
            relative_coords.append(x - base_x)
            relative_coords.append(y - base_y)
            
        max_value = max([abs(n) for n in relative_coords]) or 1
        return np.array([n / max_value for n in relative_coords])

    def extract(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        left_hand_data = np.zeros(42)
        right_hand_data = np.zeros(42)

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label
                clean_data = self.normalize_hand(hand_landmarks.landmark)
                
                if label == 'Left':
                    left_hand_data = clean_data
                else:
                    right_hand_data = clean_data

        return np.concatenate([left_hand_data, right_hand_data])
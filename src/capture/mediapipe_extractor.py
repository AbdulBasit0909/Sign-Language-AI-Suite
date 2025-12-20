import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands

class LandmarkExtractor:
    def __init__(self):
        # Configure MediaPipe to look for 2 hands
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2, 
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def normalize_hand(self, landmarks):
        """
        Converts raw landmarks to relative, normalized coordinates.
        1. Centers around wrist (index 0).
        2. Scales based on hand size (max distance).
        """
        # Extract X, Y coordinates only (Z is often noisy)
        coords = [[lm.x, lm.y] for lm in landmarks]
        
        # 1. Convert to Relative Coordinates (Center at Wrist)
        base_x, base_y = coords[0][0], coords[0][1]
        
        relative_coords = []
        for x, y in coords:
            relative_coords.append(x - base_x)
            relative_coords.append(y - base_y)
            
        # 2. Normalize Scale (Make it size-invariant)
        # Find the maximum absolute distance to scale values between -1 and 1
        max_value = max([abs(n) for n in relative_coords])
        if max_value == 0: 
            max_value = 1 # Avoid division by zero
            
        normalized_list = [n / max_value for n in relative_coords]
        
        return np.array(normalized_list)

    def extract(self, frame):
        """
        Returns a fixed-size vector of 84 features:
        [Left Hand (42) ... Right Hand (42)]
        """
        # MediaPipe works with RGB images
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        # Initialize empty arrays (filled with zeros)
        # 21 landmarks * 2 coordinates (x, y) = 42 features per hand
        left_hand_data = np.zeros(42)
        right_hand_data = np.zeros(42)

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                
                # Check label: 'Left' or 'Right'
                label = handedness.classification[0].label
                
                # Process this specific hand
                clean_data = self.normalize_hand(hand_landmarks.landmark)
                
                # Assign to the correct side
                if label == 'Left':
                    left_hand_data = clean_data
                else:
                    right_hand_data = clean_data

        # Concatenate them into one long list of numbers
        # Structure: [Left_x0, Left_y0, ... Left_y20, Right_x0, ... Right_y20]
        final_vector = np.concatenate([left_hand_data, right_hand_data])
        
        return final_vector
import os
import sys
import cv2
import numpy as np
import joblib
import pyttsx3
import threading  # <--- NEW: For background audio
from collections import deque

# --- Fix Path Imports ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(root_dir)

from src.capture.mediapipe_extractor import LandmarkExtractor

# --- Audio Helper Function (Runs in background) ---
def speak_text(text):
    """Speaks text in a new thread so video doesn't freeze/stop"""
    def _speak():
        try:
            # We initialize a local engine instance to avoid conflicts
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            pass # Ignore audio errors
    
    thread = threading.Thread(target=_speak)
    thread.start()

# --- Prediction smoother ---
class PredictionSmoother:
    def __init__(self, window_size=5):
        self.window = deque(maxlen=window_size)

    def smooth(self, label):
        self.window.append(label)
        labels, counts = np.unique(self.window, return_counts=True)
        return labels[np.argmax(counts)]

def main():
    # --- Load model ---
    model_path = os.path.join(root_dir, "model", "sign_model.pkl")
    label_path = os.path.join(root_dir, "model", "label_encoder.pkl")

    if not os.path.exists(model_path):
        print("❌ Model not found. Run 'python -m src.train.train_landmark' first!")
        return

    print("Loading model...")
    model = joblib.load(model_path)
    label_encoder = joblib.load(label_path)

    extractor = LandmarkExtractor()
    smoother = PredictionSmoother()
    cap = cv2.VideoCapture(0)

    last_spoken_word = ""
    sentence = []

    print("\n✅ Real-Time Sign Translator Started")
    print("Controls: Q=Quit | C=Clear | S=Speak Sentence")

    while True:
        ret, frame = cap.read()
        if not ret: continue

        frame = cv2.flip(frame, 1)
        landmarks = extractor.extract(frame)

        predicted_label = "..."
        confidence = 0.0

        if landmarks is not None:
            features = np.array(landmarks).reshape(1, -1)
            probs = model.predict_proba(features)[0]
            idx = np.argmax(probs)
            confidence = probs[idx]

            if confidence > 0.75:  # Slightly higher threshold for stability
                raw_label = label_encoder.inverse_transform([idx])[0]
                predicted_label = smoother.smooth(raw_label)

                # --- LOGIC: Only speak if the word CHANGED ---
                if predicted_label != last_spoken_word:
                    
                    # 1. Speak the new word
                    speak_text(predicted_label)
                    
                    # 2. Add to sentence list (prevent duplicates next to each other)
                    if not sentence or sentence[-1] != predicted_label:
                        sentence.append(predicted_label)

                    # 3. Update memory
                    last_spoken_word = predicted_label

            # Visual Feedback
            cv2.putText(frame, f"Sign: {predicted_label} ({confidence:.2f})", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            # Reset last spoken so you can say the same sign again after putting hand down
            last_spoken_word = ""
            cv2.putText(frame, "Show Hand", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show Sentence
        text_sentence = " ".join(sentence[-5:]) 
        cv2.rectangle(frame, (0, 440), (640, 480), (0, 0, 0), -1)
        cv2.putText(frame, text_sentence, (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Sign Translator AI", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'): break
        if key == ord('c'): 
            sentence = []
            last_spoken_word = ""
        if key == ord('s'):
            speak_text(" ".join(sentence))

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
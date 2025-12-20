import cv2
import numpy as np
import os
import sys

# Path fix to ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(root_dir)

from src.capture.mediapipe_extractor import LandmarkExtractor

DATA_FILE = "dataset.npy"
extractor = LandmarkExtractor()
cap = cv2.VideoCapture(0)

collected_data = []

print("\n=== AI SIGN LANGUAGE COLLECTOR (DUAL HAND) ===")
print("1. Enter Label")
print("2. Press 'S' to start BURST CAPTURE (50 samples)")
print("3. Press 'Q' to Save & Quit")

label = input("Enter label for this sign (e.g., Hello, Clap): ")

counter = 0
capturing = False

while True:
    ret, frame = cap.read()
    if not ret: continue
    
    # Flip for mirror effect
    frame = cv2.flip(frame, 1)
    
    lm = extractor.extract(frame)

    # --- THE FIX IS HERE ---
    # We check 'np.any(lm)'. 
    # If the array is all Zeros (False), it means no hand.
    # If it has numbers (True), it means a hand is there.
    hand_found = np.any(lm) 

    # UI Feedback
    if hand_found:
        color = (0, 255, 0)
        status = "Hand Detected"
    else:
        color = (0, 0, 255)
        status = "No Hand"

    cv2.putText(frame, f"Label: {label} | Samples: {len(collected_data)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, status, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    if capturing:
        cv2.putText(frame, f"CAPTURING... {counter}/50", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Only save if a hand is ACTUALLY there
        if hand_found:
            sample = np.append(lm, label)
            collected_data.append(sample)
            counter += 1
        else:
            print("⚠ specific frame skipped - no hand visible")
        
        if counter >= 50:
            capturing = False
            counter = 0
            print(f"✅ Finished collecting 50 samples for '{label}'")

    cv2.imshow("Collector", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') and not capturing:
        capturing = True
        counter = 0
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save logic
if len(collected_data) > 0:
    collected_data = np.array(collected_data, dtype=object)
    try:
        existing = np.load(DATA_FILE, allow_pickle=True)
        final_data = np.vstack([existing, collected_data])
    except:
        final_data = collected_data
    
    np.save(DATA_FILE, final_data)
    print(f"Saved. Total samples in dataset: {len(final_data)}")
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))

DATA_FILE = os.path.join(root_dir, "dataset.npy")
MODEL_PATH = os.path.join(root_dir, "model", "sign_model.pkl")
LABEL_PATH = os.path.join(root_dir, "model", "label_encoder.pkl")

def train_model_function():
    """
    Function to train the model and return accuracy.
    This allows the GUI to call it.
    """
    if not os.path.exists(DATA_FILE):
        return 0.0, "No dataset found"

    try:
        data = np.load(DATA_FILE, allow_pickle=True)
        
        # Check if data is empty
        if len(data) == 0:
            return 0.0, "Dataset empty"

        # Split features (X) and labels (y)
        # Note: Depending on collection, data might be strings or objects
        X = np.array([item[:-1] for item in data]).astype(float)
        y = np.array([item[-1] for item in data])

        # Encode labels
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        # Train Model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Save
        joblib.dump(model, MODEL_PATH)
        joblib.dump(le, LABEL_PATH)
        
        return accuracy, f"Success! Classes: {len(le.classes_)}"
        
    except Exception as e:
        return 0.0, str(e)

if __name__ == "__main__":
    acc, msg = train_model_function()
    print(f"Accuracy: {acc*100:.2f}% | {msg}")
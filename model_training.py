import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from feature_extraction import extract_features, get_file_extension

def load_data(data_dir):
    """Loads data and extracts features."""
    features = []
    labels = []
    allowed_extensions = ['.wav', '.mp3', '.flac', '.ogg']  # Add or remove extensions as needed
    for genre in os.listdir(data_dir):
        genre_path = os.path.join(data_dir, genre)
        if os.path.isdir(genre_path):
            for file in os.listdir(genre_path):
                file_path = os.path.join(genre_path, file)
                if get_file_extension(file_path) in allowed_extensions:
                    extracted_features = extract_features(file_path)
                    if extracted_features is not None:
                        features.append(extracted_features)
                        labels.append(genre)
    return pd.DataFrame(features), pd.Series(labels)

def train_model(data_dir, model_path="genre_model.joblib"):
    """Trains and saves the model."""
    X, y = load_data(data_dir)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    train_model("data")
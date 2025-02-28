import librosa
import numpy as np
import os

def extract_features(file_path):
    """Extracts audio features from a given file."""
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        print(f"Error processing file: {file_path}, error: {e}")
        return None

def get_file_extension(file_path):
    """Gets the file extension of a file."""
    _, file_extension = os.path.splitext(file_path)
    return file_extension.lower()
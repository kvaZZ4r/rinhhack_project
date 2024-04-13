import librosa
import numpy as np

def extract_audio_features(audio_file):
    # Load the audio file
    y, sr = librosa.load(audio_file)

    # Calculate features
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    return spectral_centroid, zero_crossing_rate, spectral_bandwidth, mfccs

# Example usage
audio_file = r"C:\Users\Пользователь\Downloads\test\ac9636ef0fa4.wav"
spectral_centroid, zero_crossing_rate, spectral_bandwidth, mfccs = extract_audio_features(audio_file)

# Print the shapes of the extracted features
print("Shape of spectral centroid:", spectral_centroid.shape)
print("Shape of zero-crossing rate:", zero_crossing_rate.shape)
print("Shape of spectral bandwidth:", spectral_bandwidth.shape)
print("Shape of MFCCs:", mfccs.shape)
ё
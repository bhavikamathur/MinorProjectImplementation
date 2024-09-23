import os
import librosa
import numpy as np

# Path of the dataset
dataset_path = '/Users/muditjoshi/Desktop/MinorProjectImplementation/dataset/Audio_Speech_Actors_01-24'

# Emotion labels based on the file name convention in RAVDESS
emotion_dict = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}


# Function to load all audio files and their labels
def load_ravdess_data(directory):
    all_audio_data = []
    all_labels = []

    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(subdir, file)

                # Load the audio file
                audio, sample_rate = librosa.load(file_path, sr=None)

                # Extract emotion from filename (3rd part of the filename)
                emotion = emotion_dict[file.split('-')[2]]

                # Append the audio and corresponding emotion label
                all_audio_data.append((audio, sample_rate))
                all_labels.append(emotion)

    return all_audio_data, all_labels


# Function to extract MFCC and Chroma features
def extract_features(audio_data, n_mfcc=13, n_chroma=12):
    features = []
    for audio, sample_rate in audio_data:
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        mfccs_mean = np.mean(mfccs.T, axis=0)  # Mean of MFCC features

        # Extract Chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate, n_chroma=n_chroma)
        chroma_mean = np.mean(chroma.T, axis=0)  # Mean of Chroma features

        # Concatenate MFCC and Chroma features
        combined_features = np.concatenate((mfccs_mean, chroma_mean))
        features.append(combined_features)

    return np.array(features)


# Load the dataset
audio_data, labels = load_ravdess_data(dataset_path)

# Extract both MFCC and Chroma features
combined_features = extract_features(audio_data)

# Output the results
print(f"Loaded {len(audio_data)} audio files with {len(set(labels))} unique emotions.")
print(f"Extracted combined MFCC and Chroma features with shape: {combined_features.shape}")

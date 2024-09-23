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
                all_audio_data.append(audio)
                all_labels.append(emotion)

    return all_audio_data, all_labels


# Load the dataset
audio_data, labels = load_ravdess_data(dataset_path)

# Output the results
print(f"Loaded {len(audio_data)} audio files with {len(set(labels))} unique emotions.")
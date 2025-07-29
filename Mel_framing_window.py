import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
from math import ceil

# Dataset - Crema D
cremaD = "C:/Users/visha/Desktop/SER/Data/Crema-D/AudioWAV"
directory = os.listdir(cremaD)
file_emotion = []
file_path = []

# Extract file paths and corresponding emotions
for file in directory:
    file_path.append(cremaD + "/" + file)
    part = file.split("_")
    if part[2] == 'SAD':
        file_emotion.append('sad')
    elif part[2] == 'ANG':
        file_emotion.append('angry')
    elif part[2] == 'DIS':
        file_emotion.append('disgust')
    elif part[2] == 'FEA':
        file_emotion.append('fear')
    elif part[2] == 'HAP':
        file_emotion.append('happy')
    elif part[2] == 'NEU':
        file_emotion.append('neutral')
    else:
        file_emotion.append('Unknown')

# Create dataframe for emotions and paths
emotion_df = pd.DataFrame(file_emotion, columns=["Emotions"])
path_df = pd.DataFrame(file_path, columns=["Paths"])
cremaD_df = pd.concat([emotion_df, path_df], axis=1)

# Data Visualization
plt.title("Count of emotions")
sns.countplot(x="Emotions", data=cremaD_df)
plt.ylabel("Count")
plt.xlabel("Emotions")
plt.show()

# Windowing Parameters from Analysis
WINDOW_SIZE = 31  # frames
FRAME_LENGTH = 2048  # Window size in samples
HOP_LENGTH = 512    # Hop size for overlap
TARGET_FRAMES_PER_FILE = 3.81  # From analysis

# Create output directory
flag = True  # Set to True to generate and save spectrograms

if flag:
    output_base_dir = "C:/Users/visha/Desktop/SER/Features/CremaD_Mel_Window"
    os.makedirs(output_base_dir, exist_ok=True)

    # Create directory for each emotion
    for emotion in cremaD_df['Emotions'].unique():
        os.makedirs(os.path.join(output_base_dir, emotion), exist_ok=True)

    # Generate and save spectrograms with windowing
    for i in range(len(cremaD_df)):
        # Get file path and emotion
        file_path = cremaD_df.iloc[i]['Paths']
        emotion = cremaD_df.iloc[i]['Emotions']

        # Load audio
        y, sr = librosa.load(file_path, sr=16000)
        
        # Calculate duration in seconds
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Calculate window size in seconds
        window_duration = WINDOW_SIZE * HOP_LENGTH / sr
        
        # Calculate number of windows with overlap
        n_windows = ceil(duration * TARGET_FRAMES_PER_FILE)
        
        # Calculate hop size between windows to achieve target frames per file
        if n_windows > 1:
            window_hop = (duration - window_duration) / (n_windows - 1)
        else:
            window_hop = 0
            
        # Generate spectrograms for each window
        for w in range(n_windows):
            # Calculate start time for this window
            start_time = w * window_hop
            
            # Calculate end time for this window
            end_time = start_time + window_duration
            
            # Convert times to samples
            start_sample = int(start_time * sr)
            end_sample = min(int(end_time * sr), len(y))
            
            # Extract audio segment
            y_segment = y[start_sample:end_sample]
            
            # Generate Mel spectrogram for this window
            mel_spect = librosa.feature.melspectrogram(
                y=y_segment, 
                sr=sr, 
                n_fft=FRAME_LENGTH, 
                hop_length=HOP_LENGTH
            )

            # Convert to log scale (dB)
            log_mel_spect = librosa.power_to_db(mel_spect, ref=np.max)

            # Create plot for the Mel spectrogram
            plt.figure(figsize=(6, 4))
            librosa.display.specshow(log_mel_spect, sr=sr, x_axis="time", y_axis="mel")
            plt.colorbar(format='%+2.0f dB')
            plt.title(f"Log Mel Spectrogram {i}_window_{w}")
            plt.xlabel("Time")
            plt.ylabel("Frequency (mel)")
            plt.tight_layout()

            # Save plot as image
            output_path = os.path.join(output_base_dir, emotion, f'spec_{i}_window_{w}.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

        # Print progress every 100 files
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} files out of {len(cremaD_df)}")
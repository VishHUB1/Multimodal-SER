import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil

def create_log_mel_spectrogram(audio_path, save_path, n_mels=128, n_fft=2048, hop_length=512):
    """
    Create and save log mel spectrogram image from audio file
    
    Parameters:
        audio_path (str): Path to audio file
        save_path (str): Path to save the spectrogram image
        n_mels (int): Number of mel bands
        n_fft (int): FFT window size
        hop_length (int): Hop length for STFT
    """
    # Load audio file
    y, sr = librosa.load(audio_path, sr=None)
    
    # Create mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    
    # Convert to log scale
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    
    # Create figure
    plt.figure(figsize=(10, 4))
    plt.axis('off')  # Remove axes
    
    # Display mel spectrogram
    librosa.display.specshow(
        log_mel_spectrogram, 
        sr=sr, 
        hop_length=hop_length,
        x_axis='time',
        y_axis='mel',
        cmap='viridis'
    )
    
    # Save figure without padding
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()

def process_tess_dataset(dataset_path, output_path):
    """
    Process TESS dataset to create log mel spectrograms and organize by emotion
    
    Parameters:
        dataset_path (str): Path to TESS dataset
        output_path (str): Base path to save organized spectrograms
    """
    # Create output directory if it doesn't exist
    if os.path.exists(output_path):
        shutil.rmtree(output_path)  # Remove if exists to start fresh
    os.makedirs(output_path, exist_ok=True)
    
    # Updated common TESS emotion folder names (with "surprise" instead of "ps")
    emotions = ["angry", "disgust", "fear", "happy", "neutral", "surprise", "sad"]
    
    # Standard emotion labels for our output
    emotion_map = {
        "angry": "anger",
        "disgust": "disgust",
        "fear": "fear",
        "happy": "happiness",
        "neutral": "neutral",
        "surprise": "surprise",  # Changed from "ps" to "surprise"
        "sad": "sadness"
    }
    
    # Create emotion directories for output
    for emotion in emotion_map.values():
        os.makedirs(os.path.join(output_path, emotion), exist_ok=True)
    
    # Counters
    total_files = 0
    processed_files = 0
    
    # Expected speaker directories (OAF and YAF)
    speaker_dirs = ["OAF_", "YAF_"]
    
    # Find and process all audio files
    for speaker_prefix in speaker_dirs:
        print(f"Processing {speaker_prefix.strip('_')} files...")
        
        # Look for speaker directories
        speaker_paths = []
        for item in os.listdir(dataset_path):
            if item.startswith(speaker_prefix) and os.path.isdir(os.path.join(dataset_path, item)):
                speaker_paths.append(os.path.join(dataset_path, item))
        
        # If no speaker directories found, check if there's a speaker folder containing emotion folders
        if not speaker_paths:
            speaker_folder = os.path.join(dataset_path, speaker_prefix.strip('_'))
            if os.path.isdir(speaker_folder):
                # Check for emotion folders inside speaker folder
                for emotion in emotions:
                    emotion_path = os.path.join(speaker_folder, emotion)
                    if os.path.isdir(emotion_path):
                        speaker_paths.append(emotion_path)
        
        # If still no paths found, try direct emotion directories with speaker prefix
        if not speaker_paths:
            for emotion in emotions:
                dir_path = os.path.join(dataset_path, f"{speaker_prefix}{emotion}")
                if os.path.isdir(dir_path):
                    speaker_paths.append(dir_path)
        
        if not speaker_paths:
            print(f"Warning: Could not find {speaker_prefix.strip('_')} directories")
            continue
            
        # Process each speaker directory
        for dir_path in speaker_paths:
            # Determine emotion from directory name
            dir_name = os.path.basename(dir_path).lower()
            
            emotion_found = False
            for emotion_key in emotions:
                if emotion_key in dir_name:
                    emotion = emotion_map[emotion_key]
                    emotion_found = True
                    break
            
            if not emotion_found:
                print(f"Could not determine emotion for directory {dir_path}, skipping.")
                continue
            
            # Get all wav files in the directory
            wav_files = [f for f in os.listdir(dir_path) if f.endswith('.wav')]
            total_files += len(wav_files)
            
            # Process each audio file
            for wav_file in tqdm(wav_files, desc=f"Processing {dir_name}"):
                audio_path = os.path.join(dir_path, wav_file)
                
                # Create save filename with speaker and original filename
                speaker_id = speaker_prefix.strip('_').lower()
                save_filename = f"{speaker_id}_{wav_file.replace('.wav', '.png')}"
                save_path = os.path.join(output_path, emotion, save_filename)
                
                # Create and save spectrogram
                create_log_mel_spectrogram(audio_path, save_path)
                
                # Update count
                processed_files += 1
                
                # Print progress
                if processed_files % 50 == 0:
                    print(f"Processed {processed_files}/{total_files} files ({(processed_files/total_files)*100:.1f}%)")
    
    # Print summary
    print("\nProcessing complete!")
    for emotion in emotion_map.values():
        emotion_dir = os.path.join(output_path, emotion)
        file_count = len([f for f in os.listdir(emotion_dir) if f.endswith('.png')])
        print(f"Emotion '{emotion}': {file_count} spectrograms")

if __name__ == "__main__":
    # Set paths
    dataset_path = r"C:\Users\visha\Desktop\SER\Data\Tess\TESS Toronto emotional speech set data"  # Path to TESS dataset
    output_path = "C:/Users/visha/Desktop/SER/Features/Tess_Mel"  # Where to save spectrograms
    
    # Process dataset
    process_tess_dataset(dataset_path, output_path)
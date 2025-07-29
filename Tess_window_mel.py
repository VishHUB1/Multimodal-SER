import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob

def generate_log_mel_spectrograms(dataset_path, output_path, params):
    """
    Generate log-MEL spectrogram images from TESS dataset audio files.
    
    Parameters:
    -----------
    dataset_path : str
        Path to the TESS dataset root directory
    output_path : str
        Path to save the generated spectrogram images
    params : dict
        Dictionary containing the parameters for spectrogram generation
    """
    # List of emotions in the TESS dataset
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Create emotion subdirectories
    for emotion in emotions:
        emotion_dir = os.path.join(output_path, emotion)
        os.makedirs(emotion_dir, exist_ok=True)
    
    # Counter for generated spectrograms
    total_spectrograms = 0
    
    # Process each emotion
    for emotion in emotions:
        print(f"Processing {emotion} files...")
        
        # Get all files for this emotion (both OAF and YAF)
        oaf_files = glob.glob(os.path.join(dataset_path, f"OAF_{emotion}", "*.wav"))
        yaf_files = glob.glob(os.path.join(dataset_path, f"YAF_{emotion}", "*.wav"))
        all_files = oaf_files + yaf_files
        
        # Output directory for this emotion
        emotion_dir = os.path.join(output_path, emotion)
        
        # Process each audio file
        for file_path in tqdm(all_files, desc=f"{emotion}"):
            # Extract filename without extension
            filename = os.path.basename(file_path).split('.')[0]
            
            # Add speaker prefix to filename
            if "OAF" in file_path:
                speaker = "OAF"
            else:
                speaker = "YAF"
            
            # Load audio file
            y, sr = librosa.load(file_path, sr=params['sample_rate'])
            
            # Generate MEL spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y, 
                sr=params['sample_rate'],
                n_fft=params['window_size'],
                hop_length=params['hop_length'],
                n_mels=params['n_mels'],
                fmin=params['fmin'],
                fmax=params['fmax'],
                window=params['window_function'],
                win_length=params['win_length']
            )
            
            # Convert to log scale (dB)
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Create plot
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(
                log_mel_spec,
                sr=params['sample_rate'],
                hop_length=params['hop_length'],
                x_axis='time',
                y_axis='mel',
                fmin=params['fmin'],
                fmax=params['fmax']
            )
            
            # Add color bar and title
            plt.colorbar(format='%+2.0f dB')
            plt.title(f"{speaker} - {emotion} - {filename}")
            plt.tight_layout()
            
            # Save figure
            output_filename = f"{speaker}_{filename}.png"
            output_filepath = os.path.join(emotion_dir, output_filename)
            plt.savefig(output_filepath, dpi=150)
            plt.close()
            
            total_spectrograms += 1
    
    print(f"Processing complete. Generated {total_spectrograms} log-MEL spectrogram images.")
    print(f"Output saved to: {output_path}")

def main():
    # Define paths
    dataset_path = r"C:\Users\visha\Desktop\SER\Data\Tess\TESS Toronto emotional speech set data"  # Update with your actual path
    output_path = "C:/Users/visha/Desktop/SER/Features/Tess_Mel_Window"  # Update with your desired output path
    
    # Parameters for log-MEL spectrogram generation
    params = {
        'window_size': 4096,
        'hop_length': 1024,
        'window_function': 'hann',
        'win_length': 4096,
        'sample_rate': 22050,
        'n_mels': 128,
        'fmin': 20,
        'fmax': 8000
    }
    
    # Generate log-MEL spectrograms
    generate_log_mel_spectrograms(dataset_path, output_path, params)

if __name__ == "__main__":
    main()
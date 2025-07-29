import os
import numpy as np
import librosa
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def analyze_window_sizes(dataset_path, emotions=['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad'],
                        window_sizes=[512, 1024, 2048, 4096], 
                        sr=22050, hop_ratio=0.25, n_mels=128):
    """
    Analyze different window sizes to find the optimal one for emotion classification
    in the TESS dataset.
    
    Parameters:
    -----------
    dataset_path : str
        Path to the TESS dataset root directory
    emotions : list
        List of emotions to analyze
    window_sizes : list
        List of window sizes (n_fft) to test
    sr : int
        Sampling rate
    hop_ratio : float
        Ratio of hop_length to window_size (e.g., 0.25 for 75% overlap)
    n_mels : int
        Number of MEL bands
        
    Returns:
    --------
    dict
        Dictionary of evaluation metrics for each window size
    """
    results = {}
    
    # Sample a subset of files for analysis (to save computation time)
    sample_files = []
    for emotion in emotions:
        # Get files for both OAF and YAF
        oaf_files = glob.glob(os.path.join(dataset_path, f"OAF_{emotion}", "*.wav"))
        yaf_files = glob.glob(os.path.join(dataset_path, f"YAF_{emotion}", "*.wav"))
        
        # Sample files from each category
        if oaf_files:
            sample_files.extend(np.random.choice(oaf_files, min(5, len(oaf_files)), replace=False))
        if yaf_files:
            sample_files.extend(np.random.choice(yaf_files, min(5, len(yaf_files)), replace=False))
    
    print(f"Analyzing {len(sample_files)} sample files across {len(emotions)} emotions")
    
    # Features and labels for evaluation
    all_features = {}
    all_labels = []
    
    # Extract labels for evaluation
    for file_path in sample_files:
        # Get emotion from path
        for emotion in emotions:
            if f"_{emotion}" in file_path:
                all_labels.append(emotion)
                break
    
    # Test each window size
    for window_size in window_sizes:
        print(f"Testing window size: {window_size}")
        hop_length = int(window_size * hop_ratio)
        
        features = []
        
        for file_path in tqdm(sample_files):
            # Load audio
            y, _ = librosa.load(file_path, sr=sr)
            
            # Generate MEL spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y, 
                sr=sr,
                n_fft=window_size,
                hop_length=hop_length,
                n_mels=n_mels,
                fmin=20,
                fmax=8000,
                window='hann'
            )
            
            # Convert to dB scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Extract features - use mean and std of each mel band
            mel_features = np.hstack([
                np.mean(mel_spec_db, axis=1),
                np.std(mel_spec_db, axis=1)
            ])
            
            features.append(mel_features)
        
        # Store features for this window size
        all_features[window_size] = np.array(features)
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(all_features[window_size])
        
        # Reduce dimensions for easier analysis
        pca = PCA(n_components=min(10, scaled_features.shape[1]))
        reduced_features = pca.fit_transform(scaled_features)
        
        # Evaluate separability of emotions (higher score = better separation)
        if len(np.unique(all_labels)) > 1:  # Need at least 2 clusters
            silhouette = silhouette_score(reduced_features, all_labels)
        else:
            silhouette = 0
            
        # Calculate variance explained
        variance_explained = np.sum(pca.explained_variance_ratio_)
        
        # Store results
        results[window_size] = {
            'silhouette_score': silhouette,
            'variance_explained': variance_explained,
            'pca_components': reduced_features,
            'labels': all_labels
        }
        
        print(f"Window size {window_size}: Silhouette score = {silhouette:.4f}, Variance explained = {variance_explained:.4f}")
    
    return results

def evaluate_spectral_detail(dataset_path, window_sizes=[512, 1024, 2048, 4096], sr=22050):
    """
    Evaluate spectral detail captured by different window sizes by analyzing
    frequency resolution and time-frequency tradeoff.
    
    Parameters:
    -----------
    dataset_path : str
        Path to the TESS dataset root directory
    window_sizes : list
        List of window sizes to test
    sr : int
        Sampling rate
        
    Returns:
    --------
    dict
        Dictionary of spectral analysis metrics for each window size
    """
    # Select a single file for demonstration
    all_files = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.wav'):
                all_files.append(os.path.join(root, file))
    
    # Pick a file with emotional content
    sample_file = np.random.choice(all_files)
    print(f"Analyzing spectral detail with sample file: {os.path.basename(sample_file)}")
    
    # Load audio
    y, _ = librosa.load(sample_file, sr=sr)
    
    spectral_results = {}
    
    plt.figure(figsize=(15, 10))
    
    for i, window_size in enumerate(window_sizes):
        # Calculate frequency resolution
        freq_resolution = sr / window_size
        
        # Calculate time resolution (in ms)
        time_resolution = (window_size / sr) * 1000
        
        # Generate spectrogram
        hop_length = window_size // 4  # 75% overlap
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=window_size, hop_length=hop_length,
            n_mels=128, fmin=20, fmax=8000, window='hann'
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Plot for comparison
        plt.subplot(len(window_sizes), 1, i+1)
        librosa.display.specshow(
            mel_spec_db, sr=sr, hop_length=hop_length,
            x_axis='time', y_axis='mel', fmin=20, fmax=8000
        )
        plt.title(f"Window size: {window_size} | Freq res: {freq_resolution:.1f}Hz | Time res: {time_resolution:.1f}ms")
        plt.colorbar(format='%+2.0f dB')
        
        # Store results
        spectral_results[window_size] = {
            'frequency_resolution': freq_resolution,
            'time_resolution': time_resolution,
            'time_frequency_product': freq_resolution * time_resolution
        }
    
    plt.tight_layout()
    plt.savefig('window_size_comparison.png')
    
    return spectral_results

def find_optimal_window_size(dataset_path):
    """
    Find the optimal window size by evaluating both emotion separability
    and spectral detail metrics.
    
    Parameters:
    -----------
    dataset_path : str
        Path to the TESS dataset root directory
        
    Returns:
    --------
    dict
        Dictionary of optimal parameters
    """
    # Define window sizes to test
    window_sizes = [512, 1024, 2048, 4096]
    
    # Analyze emotion separability
    separability_results = analyze_window_sizes(dataset_path, window_sizes=window_sizes)
    
    # Analyze spectral detail
    spectral_results = evaluate_spectral_detail(dataset_path, window_sizes=window_sizes)
    
    # Combine metrics
    combined_results = {}
    
    for window_size in window_sizes:
        silhouette = separability_results[window_size]['silhouette_score']
        variance = separability_results[window_size]['variance_explained']
        
        # Spectral metrics
        freq_res = spectral_results[window_size]['frequency_resolution']
        time_res = spectral_results[window_size]['time_resolution']
        
        # Score based on emotion separability (higher is better)
        emotion_score = silhouette * 0.6 + variance * 0.4
        
        # Score based on spectral detail (lower product is better in general for speech)
        # For speech emotion, we want to balance time and frequency resolution
        spectral_score = 1.0 / (spectral_results[window_size]['time_frequency_product'] / 1000)
        
        # Combined score (weighted sum)
        # For emotion recognition, we weight emotion separability higher
        combined_score = emotion_score * 0.7 + spectral_score * 0.3
        
        combined_results[window_size] = {
            'emotion_score': emotion_score,
            'spectral_score': spectral_score,
            'combined_score': combined_score,
            'silhouette': silhouette,
            'variance': variance,
            'freq_resolution': freq_res,
            'time_resolution': time_res
        }
        
        print(f"Window size {window_size}: Combined score = {combined_score:.4f}")
    
    # Find optimal window size
    optimal_window_size = max(combined_results, key=lambda k: combined_results[k]['combined_score'])
    
    print(f"\nOptimal window size: {optimal_window_size}")
    print(f"Frequency resolution: {combined_results[optimal_window_size]['freq_resolution']:.2f} Hz")
    print(f"Time resolution: {combined_results[optimal_window_size]['time_resolution']:.2f} ms")
    
    # Create recommended parameters
    hop_length = optimal_window_size // 4  # 75% overlap for better time resolution
    
    optimal_params = {
        'window_size (n_fft)': optimal_window_size,
        'hop_length': hop_length,
        'window_function': 'hann',
        'win_length': optimal_window_size,
        'sample_rate': 22050,
        'n_mels': 128,
        'fmin': 20,
        'fmax': 8000
    }
    
    print("\nRecommended Parameters:")
    for param, value in optimal_params.items():
        print(f"  {param}: {value}")
    
    return optimal_params

if __name__ == "__main__":
    # Path to the TESS dataset
    dataset_path = r"C:\Users\visha\Desktop\SER\Data\Tess\TESS Toronto emotional speech set data"  # Update with your actual path
    
    # Find optimal window size
    optimal_params = find_optimal_window_size(dataset_path)
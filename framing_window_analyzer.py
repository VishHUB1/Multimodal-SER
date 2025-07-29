import os
import numpy as np
import librosa
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical

class WindowAnalyzer:
    def __init__(self, data_path, sr=16000, n_mels=128, hop_length=512):
        self.data_path = data_path
        self.sr = sr
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.emotions = {'ANG': 0, 'DIS': 1, 'FEA': 2, 'HAP': 3, 'NEU': 4, 'SAD': 5}
    
    def load_audio(self, file_path):
        """Load and preprocess audio file"""
        audio, _ = librosa.load(file_path, sr=self.sr)
        return audio

    def create_mel_spectrogram(self, audio):
        """Create mel spectrogram from audio"""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_mels=self.n_mels,
            hop_length=self.hop_length
        )
        log_mel_spec = librosa.power_to_db(mel_spec)
        return log_mel_spec

    def frame_spectrogram(self, spec, frame_size):
        """Create frames from spectrogram"""
        frames = []
        for i in range(0, spec.shape[1] - frame_size + 1, frame_size // 2):  # 50% overlap
            frame = spec[:, i:i + frame_size]
            if frame.shape[1] == frame_size:  # Only keep complete frames
                frames.append(frame)
        return np.array(frames) if frames else np.array([])

    def analyze_durations(self):
        """Analyze audio durations in dataset"""
        print("Analyzing audio durations...")
        durations = []
        total_files = len([f for f in os.listdir(self.data_path) if f.endswith('.wav')])
        processed_files = 0

        for file in os.listdir(self.data_path):
            if file.endswith('.wav'):
                try:
                    audio = self.load_audio(os.path.join(self.data_path, file))
                    spec = self.create_mel_spectrogram(audio)
                    durations.append(spec.shape[1])
                    
                    processed_files += 1
                    if processed_files % 100 == 0:
                        print(f"Processed {processed_files}/{total_files} files")
                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")
        
        mean_dur = np.mean(durations)
        stats = {
            'mean': mean_dur,
            'std': np.std(durations),
            'min': np.min(durations),
            'max': np.max(durations),
            'suggested_windows': [
                int(mean_dur * 0.15),  # Small window
                int(mean_dur * 0.25),  # Medium window
                int(mean_dur * 0.4)    # Large window
            ]
        }
        return stats

    def prepare_data(self, window_size):
        """Prepare dataset with specific window size"""
        X, y = [], []
        total_files = len([f for f in os.listdir(self.data_path) if f.endswith('.wav')])
        processed_files = 0
        
        print(f"\nPreparing data with window size {window_size}")
        
        for file in os.listdir(self.data_path):
            if file.endswith('.wav'):
                try:
                    emotion = file.split('_')[2]
                    if emotion in self.emotions:
                        audio = self.load_audio(os.path.join(self.data_path, file))
                        spec = self.create_mel_spectrogram(audio)
                        frames = self.frame_spectrogram(spec, window_size)
                        
                        if len(frames) > 0:
                            X.extend(frames)
                            y.extend([self.emotions[emotion]] * len(frames))
                    
                    processed_files += 1
                    if processed_files % 100 == 0:
                        print(f"Processed {processed_files}/{total_files} files")
                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")
        
        X = np.array(X)
        y = np.array(y)
        
        if len(X) == 0:
            raise ValueError("No valid frames were created with the given window size")
        
        # Reshape for CNN input (samples, height, width, channels)
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        
        # Normalize
        X = (X - X.mean()) / X.std()
        
        return X, y

    def create_model(self, input_shape):
        """Create a simple CNN model"""
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(len(self.emotions), activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        return model

    def validate_window_size(self, window_size, epochs=10, batch_size=32):
        """Validate specific window size"""
        print(f"\nValidating window size: {window_size}")
        
        try:
            # Prepare data
            X, y = self.prepare_data(window_size)
            y = to_categorical(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Create and train model
            model = self.create_model(input_shape=(X.shape[1], X.shape[2], 1))
            history = model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_test, y_test),
                verbose=1
            )
            
            # Evaluate
            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
            
            # Calculate validation stability
            val_accuracies = history.history['val_accuracy']
            stability_score = 1 - np.std(val_accuracies[-5:])  # Use last 5 epochs
            
            return {
                'window_size': window_size,
                'test_accuracy': test_acc,
                'test_loss': test_loss,
                'stability_score': stability_score,
                'history': history.history,
                'frames_per_file': len(X) / len([f for f in os.listdir(self.data_path) if f.endswith('.wav')])
            }
        except Exception as e:
            print(f"Error validating window size {window_size}: {str(e)}")
            return None

    def find_optimal_window(self, results):
        """Find optimal window size based on multiple criteria"""
        # Filter out None results
        results = [r for r in results if r is not None]
        
        if not results:
            raise ValueError("No valid results to analyze")
        
        weights = {
            'accuracy': 0.5,
            'stability': 0.3,
            'frames': 0.2
        }
        
        # Normalize metrics
        max_acc = max(r['test_accuracy'] for r in results)
        max_stability = max(r['stability_score'] for r in results)
        frames_per_file = [r['frames_per_file'] for r in results]
        optimal_frames = np.median(frames_per_file)
        
        scores = []
        for result in results:
            acc_score = result['test_accuracy'] / max_acc
            stability_score = result['stability_score'] / max_stability
            frames_score = 1 - abs(result['frames_per_file'] - optimal_frames) / optimal_frames
            
            total_score = (
                weights['accuracy'] * acc_score +
                weights['stability'] * stability_score +
                weights['frames'] * frames_score
            )
            
            scores.append({
                'window_size': result['window_size'],
                'total_score': total_score,
                'metrics': {
                    'accuracy': result['test_accuracy'],
                    'stability': result['stability_score'],
                    'frames_per_file': result['frames_per_file']
                }
            })
        
        return max(scores, key=lambda x: x['total_score'])

def main():
    # Initialize analyzer
    analyzer = WindowAnalyzer(data_path='path_to_crema_d_audio')
    
    try:
        # Analyze durations
        duration_stats = analyzer.analyze_durations()
        print("\nDataset Statistics:")
        print(f"Mean duration: {duration_stats['mean']:.2f} frames")
        print(f"Min duration: {duration_stats['min']} frames")
        print(f"Max duration: {duration_stats['max']} frames")
        print("\nSuggested window sizes:", duration_stats['suggested_windows'])
        
        # Validate different window sizes
        results = []
        for window_size in duration_stats['suggested_windows']:
            result = analyzer.validate_window_size(window_size)
            if result is not None:
                results.append(result)
        
        if not results:
            raise ValueError("No valid results obtained from window size validation")
        
        # Find and print optimal window size
        optimal = analyzer.find_optimal_window(results)
        
        print("\n=== OPTIMAL WINDOW SIZE ANALYSIS ===")
        print(f"\nOptimal Window Size: {optimal['window_size']} frames")
        print(f"Overall Score: {optimal['total_score']:.4f}")
        print("\nDetailed Metrics:")
        print(f"- Accuracy: {optimal['metrics']['accuracy']:.4f}")
        print(f"- Stability: {optimal['metrics']['stability']:.4f}")
        print(f"- Frames per file: {optimal['metrics']['frames_per_file']:.2f}")
        print("\nTime duration:", (optimal['window_size'] * analyzer.hop_length / analyzer.sr), "seconds")
        
        # Print comparison with other windows
        print("\nComparison with all tested windows:")
        for result in results:
            print(f"\nWindow Size: {result['window_size']}")
            print(f"- Test Accuracy: {result['test_accuracy']:.4f}")
            print(f"- Stability Score: {result['stability_score']:.4f}")
            print(f"- Frames per file: {result['frames_per_file']:.2f}")
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()
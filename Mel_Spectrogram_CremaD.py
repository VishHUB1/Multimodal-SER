#Libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display


#Dataset - Crema D
cremaD="C:/Users/visha/Desktop/Academics/Projects/SER/Data/Crema-D/AudioWAV"
directory=os.listdir(cremaD)
file_emotion=[]
file_path=[]
for file in directory:
    file_path.append(cremaD+"/"+file)
    part=file.split("_") #1001_DFA_ANG_XX will have four splits
    if part[2] == 'SAD': #choosing 3rd split for emotion
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
emotion_df=pd.DataFrame(file_emotion,columns=["Emotions"])
path_df=pd.DataFrame(file_path,columns=["Paths"])
cremaD_df=pd.concat([emotion_df,path_df],axis=1)
print(cremaD_df.head())
#Data Visualization
plt.title("Count of emotions")
sns.countplot(x="Emotions",data=cremaD_df)
plt.ylabel("Count")
plt.xlabel("Emotions")
plt.show()

#log mel spectrogram 
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
flag=False
# Create output directory
if flag== True:
    output_base_dir = "C:/Users/visha/Desktop/Academics/Projects/SER/Data/CremaD_Mel"
    os.makedirs(output_base_dir, exist_ok=True)

# Create directory for each emotion
    for emotion in cremaD_df['Emotions'].unique():
        os.makedirs(os.path.join(output_base_dir, emotion), exist_ok=True)

# Generate and save spectrograms
    for i in range(len(cremaD_df)):
    # Get file path and emotion
        file_path = cremaD_df.iloc[i]['Paths']
        emotion = cremaD_df.iloc[i]['Emotions']
    
    # Load audio
        y, sr = librosa.load(file_path, sr=16000)
    
    # Generate mel spectrogram
        mel_spect = librosa.feature.melspectrogram(y=y, sr=sr)
        log_mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    
    # Create plot
        plt.figure(figsize=(6, 4))
        librosa.display.specshow(log_mel_spect, sr=sr, x_axis="time", y_axis="mel")
        plt.colorbar(format='%+2.0f dB')
        plt.title(f"Log mel Spectrogram {i}")
        plt.xlabel("Time")
        plt.ylabel("Frequency(mel)")
        plt.tight_layout()
    
    # Save plot
        output_path = os.path.join(output_base_dir, emotion, f'spec_{i}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
    
    # Print progress every 100 files
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} files out of {len(cremaD_df)}")
        
        

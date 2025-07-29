import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image

class EmotionDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Use PIL to open image and convert to RGB
        image = Image.open(self.image_paths[idx]).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, self.labels[idx]

def load_framed_emotion_dataset(root_dir):
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']
    image_paths, labels = [], []

    for emotion_idx, emotion in enumerate(emotions):
        emotion_path = os.path.join(root_dir, emotion)
        for img_file in os.listdir(emotion_path):
            # Full path to the image
            full_path = os.path.join(emotion_path, img_file)
            image_paths.append(full_path)
            labels.append(emotion_idx)

    return image_paths, labels

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, results_dir=None):
    train_losses, val_losses = [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            # Inception v3 returns tuple of outputs when training
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Get the main output
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}: '
              f'Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.2f}%, '
              f'Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.2f}%')
    
    if results_dir:
        # Save the model
        torch.save(model.state_dict(), os.path.join(results_dir, 'inception_framed_model.pth'))
    
    return train_losses, val_losses

def plot_learning_curves(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Learning Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(save_path)
    plt.close()

def emotion_classification_with_framed_spectrograms(
    root_dir, 
    results_dir,
    batch_size=32, 
    learning_rate=0.001, 
    num_epochs=10
):
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Load dataset
    image_paths, labels = load_framed_emotion_dataset(root_dir)
    print(f"Total images loaded: {len(image_paths)}")
    
    # Check data distribution
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']
    label_counts = {emotion: 0 for emotion in emotions}
    for label in labels:
        label_counts[emotions[label]] += 1
    
    print("Data distribution:")
    for emotion, count in label_counts.items():
        print(f"{emotion}: {count} images")

    # Split dataset
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    print(f"Training set: {len(train_paths)} images")
    print(f"Test set: {len(test_paths)} images")

    # Transformations - Inception v3 requires 299x299 input size
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # Inception v3 specific size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets and loaders
    train_dataset = EmotionDataset(train_paths, train_labels, transform)
    test_dataset = EmotionDataset(test_paths, test_labels, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)

    # Model setup - using Inception v3
    model = torchvision.models.inception_v3(pretrained=True)
    # Modify the final classifier
    model.fc = torch.nn.Linear(model.fc.in_features, 6)  # 6 emotion classes
    # Set auxiliary classifier
    model.AuxLogits.fc = torch.nn.Linear(model.AuxLogits.fc.in_features, 6)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training
    train_losses, val_losses = train_model(
        model, train_loader, test_loader, criterion, optimizer, 
        num_epochs=num_epochs, results_dir=results_dir
    )
    
    # Plot and save learning curves
    plot_learning_curves(
        train_losses, val_losses, 
        save_path=os.path.join(results_dir, 'learning_curves_framed.png')
    )

    # Evaluation
    model.eval()
    all_preds, all_labels = [], []
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Classification Report
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']
    report = classification_report(all_labels, all_preds, target_names=emotions)
    print("\nClassification Report:")
    print(report)
    
    # Save classification report
    with open(os.path.join(results_dir, 'classification_report_framed.txt'), 'w') as f:
        f.write(report)

    # Confusion Matrix
    plot_confusion_matrix(
        all_labels, 
        all_preds, 
        emotions, 
        save_path=os.path.join(results_dir, 'confusion_matrix_framed.png')
    )

    return model

# Example usage
if __name__ == '__main__':
    model = emotion_classification_with_framed_spectrograms(
        root_dir=r"C:\Users\visha\Desktop\SER\Features\CremaD_Mel_Window",
        results_dir=r"C:\Users\visha\Desktop\SER\Results_mel\Inception_framing",
        batch_size=32,
        learning_rate=0.001,
        num_epochs=15
    )
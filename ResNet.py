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

def load_emotion_dataset(root_dir):
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

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    train_losses, val_losses = [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')
    
    return train_losses, val_losses

def plot_learning_curves(train_losses, val_losses, save_path='learning_curves.png'):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Learning Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, save_path='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(save_path)
    plt.close()

def emotion_classification(
    root_dir, 
    batch_size=32, 
    learning_rate=0.001, 
    num_epochs=10,
    learning_curves_path='learning_curves.png', 
    confusion_matrix_path='confusion_matrix.png', 
    classification_report_path='classification_report.txt'
):
    # Load dataset
    image_paths, labels = load_emotion_dataset(root_dir)

    # Split dataset
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=0.3, random_state=42
    )

    # Transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets and loaders
    train_dataset = EmotionDataset(train_paths, train_labels, transform)
    test_dataset = EmotionDataset(test_paths, test_labels, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Model setup
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 6)  # 6 emotion classes
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training
    train_losses, val_losses = train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs)
    plot_learning_curves(train_losses, val_losses, save_path=os.path.join(learning_curves_path, 'learning_curves.png'))

    # Evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(model.fc.weight.device)
            labels = labels.to(model.fc.weight.device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Classification Report
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']
    report = classification_report(all_labels, all_preds, target_names=emotions)
    print(report)
    
    # Save classification report
    with open(os.path.join(classification_report_path, 'classification_report.txt'), 'w') as f:
        f.write(report)

    # Confusion Matrix
    plot_confusion_matrix(
        all_labels, 
        all_preds, 
        emotions, 
        save_path=os.path.join(confusion_matrix_path, 'confusion_matrix.png')
    )

    return model

# Example usage
if __name__ == '__main__':
    model = emotion_classification(
        root_dir="C:/Users/visha/Desktop/Academics/Projects/SER/Data/CremaD_Mel",
        learning_curves_path=r"C:\Users\visha\Desktop\Academics\Projects\SER\Results_mel\ResNet",
        confusion_matrix_path=r"C:\Users\visha\Desktop\Academics\Projects\SER\Results_mel\ResNet",
        classification_report_path=r"C:\Users\visha\Desktop\Academics\Projects\SER\Results_mel\ResNet"
    )
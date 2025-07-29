import os
import torch
import timm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler

class EmotionDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
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
            full_path = os.path.join(emotion_path, img_file)
            image_paths.append(full_path)
            labels.append(emotion_idx)

    return image_paths, labels

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10):
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience = 5  # Early stopping patience
    counter = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Use mixed precision training
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # L2 regularization
                l2_lambda = 0.001
                l2_reg = torch.tensor(0.).to(device)
                for param in model.parameters():
                    l2_reg += torch.norm(param)
                loss += l2_lambda * l2_reg
            
            # Scale loss and perform backward pass
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step with scaler
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                model.load_state_dict(torch.load('best_model.pth'))
                break
    
    return train_losses, val_losses

def emotion_classification(
    root_dir, 
    batch_size=16,  # Reduced batch size for ViT
    learning_rate=1e-4,  # Reduced learning rate for ViT
    num_epochs=30,
    learning_curves_path='learning_curves.png', 
    confusion_matrix_path='confusion_matrix.png', 
    classification_report_path='classification_report.txt'
):
    # Load dataset
    image_paths, labels = load_emotion_dataset(root_dir)

    # Split dataset
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        val_paths, val_labels, test_size=0.5, random_state=42
    )

    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ViT standard size
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Validation/Test transforms
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = EmotionDataset(train_paths, train_labels, train_transform)
    val_dataset = EmotionDataset(val_paths, val_labels, val_transform)
    test_dataset = EmotionDataset(test_paths, test_labels, val_transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    # Model setup - using ViT
    model = timm.create_model('vit_large_patch14_clip_224', pretrained=True, num_classes=6)
    
    # Custom head for emotion classification
    num_features = model.head.in_features
    model.head = torch.nn.Sequential(
        torch.nn.LayerNorm(num_features),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(num_features, 1024),
        torch.nn.GELU(),
        torch.nn.Dropout(p=0.3),
        torch.nn.Linear(1024, 6)
    )
    
    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.05,  # Increased weight decay for ViT
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=3,
        verbose=True,
        min_lr=1e-6
    )

    # Training
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs
    )
    
    # Save learning curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Learning Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(learning_curves_path, 'learning_curves.png'))
    plt.close()

    # Evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(next(model.parameters()).device)
            labels = labels.to(next(model.parameters()).device)
            with autocast():
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
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=emotions, yticklabels=emotions)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(confusion_matrix_path, 'confusion_matrix.png'))
    plt.close()

    return model

if __name__ == '__main__':
    model = emotion_classification(
        root_dir=r"C:\Users\visha\Desktop\SER\Features\CremaD_Mel",
        learning_curves_path=r"C:\Users\visha\Desktop\SER\Results_mel\ViT",
        confusion_matrix_path=r"C:\Users\visha\Desktop\SER\Results_mel\ViT",
        classification_report_path=r"C:\Users\visha\Desktop\SER\Results_mel\ViT"
    )
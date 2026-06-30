import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import vgg16, VGG16_Weights
from sklearn.model_selection import train_test_split
from src.preprocessing import preprocess_image
from src.evaluation import evaluate_model

# 1. Simple In-Memory Dataset (repeats grayscale to 3 channels for VGG16)
class FingerprintDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def load_dataset_records(dataset_dir="dataset_blood_group"):
    blood_groups = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]
    image_records = []
    
    for bg in blood_groups:
        bg_folder = os.path.join(dataset_dir, bg)
        if os.path.exists(bg_folder) and os.path.isdir(bg_folder):
            files = [f for f in os.listdir(bg_folder) if f.lower().endswith(".bmp")]
            for filename in files:
                image_records.append({
                    "path": os.path.join(bg_folder, filename),
                    "blood_group": bg
                })
    return image_records

def preload_and_preprocess_all(image_records, target_size=(64, 64)):
    """
    Preloads and pre-processes all fingerprint images, resizing to 64x64
    and duplicating the grayscale channel to 3 channels to make it compatible
    with the pre-trained VGG16 model.
    """
    print(f"Preloading and preprocessing {len(image_records)} images...")
    images = []
    labels = []
    
    blood_groups = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]
    label_map = {bg: idx for idx, bg in enumerate(blood_groups)}
    
    for i, record in enumerate(image_records):
        img_path = record["path"]
        label_str = record["blood_group"]
        
        # 1. Apply Step 2 Preprocessing
        img = preprocess_image(img_path)
        if img is not None:
            # 2. Resize to 64x64 for fast CPU computation
            img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            
            # Convert to 3-channel PyTorch Tensor: (3, 64, 64)
            img_tensor = torch.tensor(img_resized, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1)
            images.append(img_tensor)
            labels.append(label_map[label_str])
            
        if (i + 1) % 1000 == 0 or (i + 1) == len(image_records):
            print(f"   Loaded and preprocessed {i + 1}/{len(image_records)} images...")
            
    return torch.stack(images), torch.tensor(labels, dtype=torch.long)

def main():
    dataset_dir = "dataset_blood_group"
    models_dir = "models"
    epochs = 5
    batch_size = 64
    learning_rate = 0.001

    print("=== Training VGG16 CNN Model (PyTorch) ===")
    
    # Check if GPU is available (fallback to CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load records
    image_records = load_dataset_records(dataset_dir)
    if not image_records:
        print(f"Error: No images found in dataset directory '{dataset_dir}'.")
        return

    # 2. Preload and Preprocess images to memory
    images, labels = preload_and_preprocess_all(image_records)

    # 3. Train-Test Split (80% train, 20% test)
    print("Splitting dataset into train (80%) and test (20%) sets...")
    indices = np.arange(len(images))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=labels.numpy()
    )
    
    train_dataset = FingerprintDataset(images[train_idx], labels[train_idx])
    test_dataset = FingerprintDataset(images[test_idx], labels[test_idx])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"   Train DataLoader size: {len(train_loader)} batches")
    print(f"   Test DataLoader size:  {len(test_loader)} batches")

    # 4. Load VGG16 model with pre-trained weights
    print("Loading pre-trained VGG16 weights...")
    model = vgg16(weights=VGG16_Weights.DEFAULT)
    
    # Freeze all layers of VGG16 to allow fast CPU training
    print("Freezing VGG16 base parameters...")
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace the final fully connected layer for 8 classes (classifier[6] is the Linear layer)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, 8)
    
    model = model.to(device)

    # 5. Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    # Optimize only the final classification layer (model.classifier[6])
    optimizer = optim.Adam(model.classifier[6].parameters(), lr=learning_rate)

    # 6. Training Loop
    print("\nTraining VGG16 classifier...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_imgs, batch_lbls in train_loader:
            batch_imgs, batch_lbls = batch_imgs.to(device), batch_lbls.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_imgs)
            loss = criterion(outputs, batch_lbls)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_imgs.size(0)
            _, predicted = outputs.max(1)
            total += batch_lbls.size(0)
            correct += predicted.eq(batch_lbls).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"   Epoch {epoch}/{epochs} - Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f} ({epoch_acc*100:.2f}%)")

    print("VGG16 model training completed!")

    # 7. Evaluation
    print("Evaluating VGG16 model on test set...")
    model.eval()
    all_preds = []
    all_targets = []
    
    blood_groups = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]
    
    with torch.no_grad():
        for batch_imgs, batch_lbls in test_loader:
            batch_imgs = batch_imgs.to(device)
            outputs = model(batch_imgs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(batch_lbls.numpy())

    # Map integer labels back to blood group names for evaluate_model
    y_test_str = np.array([blood_groups[idx] for idx in all_targets])
    y_pred_str = np.array([blood_groups[idx] for idx in all_preds])

    evaluate_model(y_test_str, y_pred_str, model_name="VGG16")

    # 8. Save model state dict
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "cnn_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Successfully saved VGG16 model weights to: {model_path}")

if __name__ == "__main__":
    main()

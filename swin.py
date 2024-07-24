import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image
from tqdm import tqdm
from torchvision.models.video import swin3d_t
from torch.cuda.amp import GradScaler, autocast

# Constants
DATA_DIR = 'data/raw_frames'
ANNOTATIONS_DIR = 'data/cvb_in_ava_format'
TRAIN_CSV = 'ava_train_set.csv'
VAL_CSV = 'ava_val_set.csv'
IMG_HEIGHT, IMG_WIDTH = 224, 224
NUM_CLASSES = 11
BATCH_SIZE = 2  # Batch size 8
ACCUMULATION_STEPS = 2  # Accumulating gradients over 2 steps
EPOCHS = 1
MODEL_SAVE_PATH = 'swin_transformer_model.pth'

class VideoDataset(Dataset):
    def __init__(self, annotations_file, transform=None):
        self.annotations = pd.read_csv(
            annotations_file,
            skiprows=1,
            names=[
                "video_name",
                "keyframe",
                "x1",
                "y1",
                "x2",
                "y2",
                "behavior_category",
                "animal_category",
            ],
            dtype={
                "keyframe": float,
                "x1": float,
                "y1": float,
                "x2": float,
                "y2": float,
                "behavior_category": int,
                "animal_category": str,
            },
            low_memory=False,
        )
        self.transform = transform

        # Debug: Check label range
        min_label = self.annotations['behavior_category'].min()
        max_label = self.annotations['behavior_category'].max()
        print(f"Label range: {min_label} to {max_label}")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        video_name = self.annotations.iloc[idx, 0]
        frames = []

        for i in range(450):
            frame_path = os.path.join(DATA_DIR, video_name, f'img_{i+1:05d}.jpg')
            if os.path.exists(frame_path):
                image = read_image(frame_path).float() / 255.0  # Normalize to [0, 1]
                if self.transform:
                    image = self.transform(image)
                frames.append(image)
            else:
                # Append a zero frame if the frame file doesn't exist
                frames.append(torch.zeros((3, IMG_HEIGHT, IMG_WIDTH)))

        clip = torch.stack(frames)  # Shape: [450, 3, 224, 224]
        clip = clip.permute(1, 0, 2, 3)  # Change to [3, 450, 224, 224]
        label = int(self.annotations.iloc[idx, 6]) - 2  # Adjust 2-based to 0-based index

        # Debug: Check label validity
        if label < 0 or label >= NUM_CLASSES:
            print(f"Invalid label {label} at index {idx}")

        return clip, label

# Data transforms
transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to load datasets
def load_datasets():
    train_dataset = VideoDataset(os.path.join(ANNOTATIONS_DIR, TRAIN_CSV), transform=transform)
    val_dataset = VideoDataset(os.path.join(ANNOTATIONS_DIR, VAL_CSV), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    return train_loader, val_loader

# Define the Swin Transformer model
class SwinTransformerModel(nn.Module):
    def __init__(self, num_classes):
        super(SwinTransformerModel, self).__init__()
        self.model = swin3d_t(weights=None)
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load datasets
    train_loader, val_loader = load_datasets()

    # Initialize the model, loss function, and optimizer
    model = SwinTransformerModel(num_classes=NUM_CLASSES).to(device)
    
    # Wrap the model with DataParallel
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Initialize GradScaler for mixed precision training
    scaler = GradScaler()
    
    torch.cuda.empty_cache()

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        optimizer.zero_grad()

        for i, (clips, labels) in enumerate(tqdm(train_loader)):
            if clips is None:
                continue

            clips, labels = clips.to(device), labels.to(device)
            clips = clips.permute(0, 1, 2, 3, 4)  # Ensure [batch_size, channels, num_frames, height, width]

            with autocast():
                outputs = model(clips)
                loss = criterion(outputs, labels)
                loss = loss / ACCUMULATION_STEPS  # Normalize loss to account for gradient accumulation

            scaler.scale(loss).backward()

            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * ACCUMULATION_STEPS

            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # Print training metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100. * correct / total
        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

        # Validation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for clips, labels in tqdm(val_loader):
                if clips is None:
                    continue

                clips, labels = clips.to(device), labels.to(device)
                clips = clips.permute(0, 1, 2, 3, 4)  # Ensure [batch_size, channels, num_frames, height, width]

                with autocast():
                    outputs = model(clips)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()

                # Calculate validation accuracy
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        # Print validation metrics
        val_epoch_loss = val_loss / len(val_loader)
        val_epoch_accuracy = 100. * correct / total
        print(f'Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_accuracy:.2f}%')

    # Save the model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f'Model saved to {MODEL_SAVE_PATH}')

if __name__ == '__main__':
    main()

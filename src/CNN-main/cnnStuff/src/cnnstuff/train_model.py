import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import os
from .audio_model import AudioCNN, audio_to_spectrogram

class SimpleAudioDataset(Dataset):
    def __init__(self, labels_file, audio_dir, all_labels):
        with open(labels_file, 'r') as f:
            self.labels_data = json.load(f)
        self.audio_dir = audio_dir
        self.files = list(self.labels_data.keys())
        
        # Create a mapping from label string to index
        self.label_to_idx = {label: i for i, label in enumerate(all_labels)}
        self.num_classes = len(all_labels)
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        filename = self.files[idx]
        # Get list of labels for this file
        labels_str = self.labels_data[filename]
        
        # Create multi-hot encoded vector
        label_tensor = torch.zeros(self.num_classes)
        for label in labels_str:
            label_idx = self.label_to_idx[label]
            label_tensor[label_idx] = 1.0
        
        # Convert audio to spectrogram
        audio_path = os.path.join(self.audio_dir, filename)
        spectrogram = audio_to_spectrogram(audio_path)
        
        return spectrogram, label_tensor

def train_simple_model(epochs=20):
    # Dynamically get the project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
    data_dir = os.path.join(project_root, 'data')

    labels_json_path = os.path.join(data_dir, "labels.json")
    manual_labels_json_path = os.path.join(data_dir, "manual_labels.json")
    audio_chunks_dir = os.path.join(data_dir, "audio_chunks")

    # Load all possible labels from the config file
    try:
        with open(labels_json_path, "r") as f:
            all_labels = json.load(f)
    except FileNotFoundError:
        print(f"Error: `{labels_json_path}` not found. Please create it.")
        return

    # Create dataset
    dataset = SimpleAudioDataset(manual_labels_json_path, audio_chunks_dir, all_labels)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Create model
    model = AudioCNN(num_classes=len(all_labels))
    # Use BCEWithLogitsLoss for multi-label classification
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}')
    
    # Save model
    torch.save(model.state_dict(), os.path.join(project_root, 'audio_model.pth'))
    print(f"Model saved as {os.path.join(project_root, 'audio_model.pth')}")
    
    return model

if __name__ == "__main__":
    train_simple_model()
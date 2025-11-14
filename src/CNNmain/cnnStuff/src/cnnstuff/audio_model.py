import torch
import torch.nn as nn
import torchaudio.transforms as T
import librosa
import numpy as np

class AudioCNN(nn.Module):
    """Simple CNN for audio classification"""
    def __init__(self, num_classes=7): # Adjusted for new label count
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), 
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        # No activation here, BCEWithLogitsLoss will handle it
        return self.classifier(x)

def audio_to_spectrogram(audio_path):
    """Convert audio file to spectrogram tensor"""
    # Load audio
    audio, sr = librosa.load(audio_path, sr=22050, duration=3.0)
    
    # Convert to tensor
    audio_tensor = torch.tensor(audio)
    
    # Create mel spectrogram
    mel_transform = T.MelSpectrogram(sample_rate=22050, n_mels=64)
    mel_spec = mel_transform(audio_tensor)
    
    # Add channel dimension for CNN
    mel_spec = mel_spec.unsqueeze(0)
    
    return mel_spec
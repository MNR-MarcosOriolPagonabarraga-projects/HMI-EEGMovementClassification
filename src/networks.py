import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGNet(nn.Module):
    def __init__(self, n_channels=21, n_classes=4, sfreq=128, 
                 F1=8, D=2, F2=16, dropout_rate=0.5):
        super(EEGNet, self).__init__()
        
        # Temporal Convolution (Input Layer)
        # Output: (Batch, F1, n_channels, samples)
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, F1, (1, sfreq // 2), padding=(0, sfreq // 4), bias=False),
            nn.BatchNorm2d(F1)
        )
        
        # Depthwise Convolution (Spatial Filtering)
        # Output: (Batch, F1*D, 1, samples // 4)
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(F1, D * F1, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(D * F1),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate)
        )
        
        # Temporal Summary (Separable Convolution)
        # Output: (Batch, F2, 1, (samples // 4) // 8)
        self.temporal_summary = nn.Sequential(
            nn.Conv2d(D * F1, D * F1, (1, 16), padding=(0, 8), groups=D * F1, bias=False),
            nn.Conv2d(D * F1, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate)
        )
        
        # Classifier
        # With 321 samples, after 4x and 8x pooling (32 total), we have ~10 points left
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(F2 * 10, n_classes) 
        )

    def forward(self, x):
        x = self.temporal_conv(x)
        x = self.depthwise_conv(x)
        x = self.temporal_summary(x)
        return self.classifier(x)


class EEGPsdNet(nn.Module):
    def __init__(self, n_channels=21, n_classes=4, sfreq=128, F1=8, D=4, F2=16):
        super(EEGPsdNet, self).__init__()
        
        # --- BRANCH 1: RAW EEG (EEGNet Encoder) ---
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, F1, (1, sfreq // 2), padding=(0, sfreq // 4), bias=False),
            nn.BatchNorm2d(F1)
        )
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(F1, D * F1, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(D * F1),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.3)
        )
        self.separator_conv = nn.Sequential(
            nn.Conv2d(D * F1, D * F1, (1, 16), padding=(0, 8), groups=D * F1, bias=False),
            nn.Conv2d(D * F1, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.3)
        )
        
        # --- BRANCH 2: PSD FEATURES ---
        # Input shape expected: (Batch, 1, 21, 28)
        self.psd_encoder = nn.Sequential(
            # 1. Spectral Convolution (Looks at local frequency bands per channel)
            # Kernel (1, 5) looks at 5Hz windows independently within each channel
            nn.Conv2d(1, 8, (1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(8),
            nn.ELU(),
            
            # 2. Spatial Convolution (Mixes all 21 channels together for each frequency band)
            nn.Conv2d(8, 16, (n_channels, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            
            # 3. Pooling and Dropout
            nn.AvgPool2d((1, 4)), # Compresses the 28 frequency bins down to 7
            nn.Dropout(0.5),
            nn.Flatten()
        )

        # --- FUSION LAYER ---
        # Raw output size: 128 (assuming temporal cropping to 256 samples)
        # PSD output size: 16 filters * 7 frequency bins = 112
        self.fusion_norm = nn.BatchNorm1d(128 + 112)
        self.classifier = nn.Linear(128 + 112, n_classes)

    def forward(self, x_raw, x_psd):
        # Branch 1
        x1 = self.temporal_conv(x_raw)
        x1 = self.spatial_conv(x1)
        x1 = self.separator_conv(x1)
        x1 = torch.flatten(x1, 1)
        
        # Branch 2
        x2 = self.psd_encoder(x_psd)
        
        # Concatenate
        combined = torch.cat((x1, x2), dim=1)
        # Normalize the combined features so one branch doesn't dominate
        combined = self.fusion_norm(combined)
        
        return self.classifier(combined)
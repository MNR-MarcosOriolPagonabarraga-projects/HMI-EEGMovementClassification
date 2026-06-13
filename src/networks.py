import torch
import torch.nn as nn

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
        # With 257 samples, after 4x and 8x pooling, we have ~8 points left
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(F2 * 8, n_classes) 
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


class EEGConnectivityNet(nn.Module):
    def __init__(self, n_channels=21, n_classes=4, sfreq=128, F1=8, D=4, F2=16):
        super(EEGConnectivityNet, self).__init__()
        
        # --- BRANCH 1: RAW CONVOLUTIONAL ENCODER ---
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
            nn.Dropout(0.4) # Increased to limit raw branch overfitting
        )
        
        # --- BRANCH 2: COMPACT CONNECTIVITY ENCODER ---
        # Squeezing the feature dimension prevents memorizing trial-level variance
        self.conn_encoder = nn.Sequential(
            nn.Linear(420, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(0.5), # High dropout forces the network to rely on broad networks
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ELU()
        )

        # --- BRANCH DIMENSION EQUALIZATION ---
        # Matches raw feature outputs (160 dimensions) down to 32 dimensions
        self.raw_compressor = nn.Sequential(
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Dropout(0.3)
        )

        # --- MULTI-MODAL ATTENTION FUSION ---
        # Determines which branch to trust on a trial-by-trial basis
        self.attention_gate = nn.Sequential(
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(16, n_classes)
        )

    def forward(self, x_raw, x_conn):
        # Process Raw Branch
        x1 = self.temporal_conv(x_raw)
        x1 = self.spatial_conv(x1)
        x1 = self.separator_conv(x1)
        x1 = torch.flatten(x1, 1) # Yields 160 dimensions
        x1_compressed = self.raw_compressor(x1) # Squeezed to 32 dimensions
        
        # Process Connectivity Branch
        x2 = self.conn_encoder(x_conn) # Yields 32 dimensions
        
        # Compute Dynamic Attention Weights
        combined_features = torch.cat((x1_compressed, x2), dim=1) # 64 dimensions
        attn_weights = self.attention_gate(combined_features)     # Shape: (Batch, 2)
        
        #  Apply Attention Weights to Blend Modalities
        w_raw = attn_weights[:, 0].unsqueeze(1)
        w_conn = attn_weights[:, 1].unsqueeze(1)
        
        fused_representation = (w_raw * x1_compressed) + (w_conn * x2)
        
        return self.classifier(fused_representation)
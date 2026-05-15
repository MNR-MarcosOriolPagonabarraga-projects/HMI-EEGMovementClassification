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

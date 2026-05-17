import numpy as np
import mne
from mne.time_frequency import csd_array_morlet

def compute_connectivity_features(epochs_data, sfreq=128):
    """
    Computes the Cross-Spectral Density (CSD) upper-triangle for each epoch.
    Input shape: (Epochs, 21 channels, Time samples)
    Output shape: (Epochs, 420 dense connectivity features)
    """
    # 10Hz captures Mu/Alpha band; 22Hz captures Beta band motor engagement
    frequencies = [10.0, 22.0]
    
    n_epochs, n_channels, n_times = epochs_data.shape
    triu_idx = np.triu_indices(n_channels, k=1)
    
    connectivity_vectors = []
    
    for i in range(n_epochs):
        # Slice out a single epoch, but keep it 3D so MNE accepts it: shape (1, 21, Time)
        single_epoch = epochs_data[i:i+1]
        
        # Compute CSD
        csd = csd_array_morlet(single_epoch, sfreq=sfreq, frequencies=frequencies, n_cycles=5, verbose=False)
        
        # Extract the exact 2D (21, 21) matrices for Alpha and Beta
        alpha_2d = np.abs(csd.get_data(index=0))
        beta_2d = np.abs(csd.get_data(index=1))
        
        # Flatten the upper triangles into 1D arrays (210 features each)
        alpha_connections = alpha_2d[triu_idx]
        beta_connections = beta_2d[triu_idx]
        
        # Concatenate to 420 features and store
        combined_vector = np.concatenate([alpha_connections, beta_connections])
        connectivity_vectors.append(combined_vector)

    # 1. Convert to numpy array safely
    connectivity_vectors = np.array(connectivity_vectors, dtype=np.float32)

    # 2. Apply Log-transform (Essential for skewed exponential spectral features)
    # Using 1e-10 ensures stability without warping the distribution threshold
    X_conn_log = np.log10(connectivity_vectors + 1e-10)
    
    # 3. Z-SCORE NORMALIZATION (Standardize features across the entire dataset)
    # This strips away the massive negative numbers and maps features to standard deviation units
    mean = np.mean(X_conn_log, axis=0, keepdims=True)
    std = np.std(X_conn_log, axis=0, keepdims=True) + 1e-8
    
    X_conn_scaled = (X_conn_log - mean) / std
        
    return X_conn_scaled.astype(np.float32)
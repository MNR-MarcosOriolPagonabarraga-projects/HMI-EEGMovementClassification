import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

from src.utils import load_npz_split, plot_history
from src.load_data import EEGPsdDataset
from src.networks import EEGPsdNet

train_dataset = "data/processed/dataset_train.npz"
test_dataset = "data/processed/dataset_test.npz"
output_folder = "models/eeg_psd"

def main():
    # Ensure our output directory structure exists
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    X_train, X_psd_train, y_train, _ = load_npz_split(train_dataset)
    X_test, X_psd_test, y_test, _ = load_npz_split(test_dataset)
    train_loader, test_loader = _create_loaders(X_train, X_psd_train, y_train, X_test, X_psd_test, y_test)

    # Build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEGPsdNet(n_channels=21, n_classes=4, sfreq=128).to(device)

    # Get class weights
    _, counts = np.unique(y_train, return_counts=True)
    class_weights = 1.0 / torch.tensor(counts, dtype=torch.float)
    class_weights = class_weights / class_weights.sum()  # Normalize
    class_weights = class_weights.to(device)

    # Train model
    num_epochs = 100
    history, all_preds, all_labels = train_loop(model, train_loader, test_loader, class_weights, num_epochs, device, out_dir)
    
    fig = plot_history(history)
    if fig is not None:
        fig.savefig(out_dir / "learning_curves.png", dpi=300)
        plt.close(fig)

    save_confusion_matrix(all_labels, all_preds, out_dir)

    print(f"\n[Artifact Storage] All training assets successfully saved to: {out_dir}/")


def _create_loaders(X_train, X_psd_train, y_train, X_test, X_psd_test, y_test):
    batch_size = 32
    train_ds = EEGPsdDataset(X_train, X_psd_train, y_train)
    test_ds = EEGPsdDataset(X_test, X_psd_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def save_confusion_matrix(labels, preds, out_dir):
    """Computes, normalizes, and saves the confusion matrix plot."""
    class_labels = ['Rest', 'Elbow', 'Hand', 'Forearm']
    cm = confusion_matrix(labels, preds)
    
    # Normalize rows to show percentages (evaluation precision)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_percent, 
        annot=True, 
        fmt=".1f", 
        cmap="Blues", 
        xticklabels=class_labels, 
        yticklabels=class_labels,
        cbar_kws={'label': 'Accuracy (%)'},
        ax=ax
    )
    ax.set_title('Validation Confusion Matrix (Optimal Weights Peak)')
    ax.set_xlabel('Predicted Pattern Target')
    ax.set_ylabel('True Pattern Target')
    
    plt.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=300)
    plt.close(fig)


def train_loop(model, train_loader, test_loader, class_weights, num_epochs, device, out_dir):
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    
    final_preds = []
    final_labels = []

    print(f"{'Epoch':<8} | {'Train Loss':<12} | {'Train Acc':<12} | {'Val Loss':<10} | {'Val Acc':<10}")
    print("-" * 65)

    for epoch in range(num_epochs):
        # --- TRAINING PHASE ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for batch_raw, batch_psd, labels in train_loader:
            batch_raw, batch_psd, labels = batch_raw.to(device), batch_psd.to(device), labels.to(device)

            if model.training:
                raw_noise = torch.randn_like(batch_raw) * 0.1
                batch_raw = batch_raw + raw_noise
                psd_noise = torch.randn_like(batch_psd) * 0.01
                batch_psd = batch_psd + psd_noise
            
            optimizer.zero_grad()
            outputs = model(batch_raw, batch_psd)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = 100. * train_correct / train_total

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        epoch_preds = []
        epoch_labels = []
        
        with torch.no_grad():
            for batch_raw, batch_psd, labels in test_loader:
                batch_raw, batch_psd, labels = batch_raw.to(device), batch_psd.to(device), labels.to(device)
                
                outputs = model(batch_raw, batch_psd)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * labels.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                epoch_preds.extend(predicted.cpu().numpy())
                epoch_labels.extend(labels.cpu().numpy())
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = 100. * val_correct / val_total
        
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        scheduler.step(epoch_val_loss)
        
        print(f"{epoch+1:<8} | {epoch_train_loss:<12.4f} | {epoch_train_acc:<10.2f}% | {epoch_val_loss:<10.4f} | {epoch_val_acc:<10.2f}%")

        # Capture the predictions exactly at the peak validation epoch
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            final_preds = epoch_preds
            final_labels = epoch_labels
            torch.save(model.state_dict(), out_dir / "EEGPsd.pth")
            print(f"New optimal validation peak encountered! Weights secured ({best_val_acc:.2f}%)")

    return history, np.array(final_preds), np.array(final_labels)

if __name__ == "__main__":
    main()
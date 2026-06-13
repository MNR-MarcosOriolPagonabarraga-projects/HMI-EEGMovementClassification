import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.networks import EEGConnectivityNet
from src.utils import plot_history, plot_and_save_latent_space, extract_dl_features
from src.load_data import EEGConnDataset


train_dataset_path = "data/processed/dataset_train.npz"
test_dataset_path = "data/processed/dataset_test.npz"
output_folder = "models/eeg_conn"

def main():
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load the new datasets containing the X_conn feature
    train_data = np.load(train_dataset_path)
    X_train_raw, X_train_conn, y_train = train_data['X'], train_data['X_conn'], train_data['y']
    
    test_data = np.load(test_dataset_path)
    X_test_raw, X_test_conn, y_test = test_data['X'], test_data['X_conn'], test_data['y']

    train_loader, test_loader = _create_loaders(X_train_raw, X_train_conn, y_train, X_test_raw, X_test_conn, y_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the new Dual-Branch Architecture
    model = EEGConnectivityNet(n_channels=21, n_classes=4, sfreq=128).to(device)

    # Handle class imbalances
    _, counts = np.unique(y_train, return_counts=True)
    class_weights = 1.0 / torch.tensor(counts, dtype=torch.float)
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(device)

    # Execute training loop
    history, all_preds, all_labels = train_loop(model, train_loader, test_loader, class_weights, device, out_dir)
    
    # Save learning curves using your util function
    fig = plot_history(history)
    if fig is not None:
        fig.savefig(out_dir / "learning_curves.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    # Save the normalized confusion matrix
    save_confusion_matrix(all_labels, all_preds, out_dir)

    print("\n[Artifact Storage] Generating t-SNE latent space visualization...")
    # 1. Load the best weights
    model.load_state_dict(torch.load(out_dir / "best_dual_branch.pth")) # or "EEGPsd.pth"
    
    # 2. Extract features (set is_dual_branch=True!)
    features, labels = extract_dl_features(model, test_loader, device, is_dual_branch=True)
    
    plot_and_save_latent_space(
        features=features, 
        labels=labels, 
        title="Dual-Branch Latent Space (Pre-decision)", 
        save_path=out_dir / "TSNE_DualBranch_Latent_Space.png", 
        is_csp=False
    )
    
    print(f"\nTraining Complete. All assets and peak weights saved to {out_dir}/")


def _create_loaders(X_train_raw, X_train_conn, y_train, X_test_raw, X_test_conn, y_test):
    batch_size = 32
    train_ds = EEGConnDataset(X_train_raw, X_train_conn, y_train)
    test_ds = EEGConnDataset(X_test_raw, X_test_conn, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def save_confusion_matrix(labels, preds, out_dir):
    class_labels = ['Rest', 'Elbow', 'Hand', 'Forearm']
    cm = confusion_matrix(labels, preds)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Blues", 
                xticklabels=class_labels, yticklabels=class_labels, ax=ax)
    ax.set_title('Validation Confusion Matrix (Optimal Weights)')
    ax.set_xlabel('Predicted Target')
    ax.set_ylabel('True Target')
    plt.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=300)
    plt.close(fig)


def train_loop(model, train_loader, test_loader, class_weights, device, out_dir):
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.03)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    num_epochs = 50 # Bumped to 50 to let the dual branches converge
    best_val_acc = 0.0
    
    final_preds, final_labels = [], []

    print(f"\n{'Epoch':<8} | {'Train Loss':<12} | {'Train Acc':<12} | {'Val Loss':<10} | {'Val Acc':<10}")
    print("-" * 65)

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        # Notice we are unpacking raw AND conn features now
        for batch_raw, batch_conn, labels in train_loader:
            batch_raw, batch_conn, labels = batch_raw.to(device), batch_conn.to(device), labels.to(device)

            if model.training:
                # Add tiny augmentation noise to prevent overfitting
                batch_raw = batch_raw + (torch.randn_like(batch_raw) * 0.1)
                batch_conn = batch_conn + (torch.randn_like(batch_conn) * 0.01)
            
            optimizer.zero_grad()
            # Pass both branches into the forward pass
            outputs = model(batch_raw, batch_conn) 
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

        # --- VALIDATION ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        epoch_preds, epoch_labels = [], []
        
        with torch.no_grad():
            for batch_raw, batch_conn, labels in test_loader:
                batch_raw, batch_conn, labels = batch_raw.to(device), batch_conn.to(device), labels.to(device)
                
                outputs = model(batch_raw, batch_conn)
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

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            final_preds = epoch_preds
            final_labels = epoch_labels
            torch.save(model.state_dict(), out_dir / "best_dual_branch.pth")
            print(f"Peak weights secured! ({best_val_acc:.2f}%)")

    return history, np.array(final_preds), np.array(final_labels)

if __name__ == "__main__":
    main()
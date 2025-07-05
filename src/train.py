# src/train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm # For a nice progress bar
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os


# Import our custom classes
from src.dataset import PhishingDataset
from src.model import PhishGuardModel

# --- Configuration ---
# Model and Data Config
TEXT_MODEL_NAME = 'distilbert-base-uncased'
VISION_MODEL_NAME = 'google/vit-base-patch16-224-in21k'
METADATA_PATH = 'data/processed/metadata.parquet'
MODEL_SAVE_PATH = 'models/phishguard_model.pt'

# Training Hyperparameters
EPOCHS = 5
BATCH_SIZE = 16 # Adjust based on your VRAM
LEARNING_RATE = 1e-5 # A smaller learning rate is good for fine-tuning
TRAIN_VAL_SPLIT_RATIO = 0.9

# Ensure the models directory exists
os.makedirs('models', exist_ok=True)

def train_model():
    """Main function to run the training and validation loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load Data ---
    # Identify structured feature columns from the parquet file
    # We exclude identifier, text, path and label columns
    structured_cols = [
    'url_len', 
    'hostname_len', 
    'path_len', 
    'num_dots', 
    'has_ip', 
    'domain_age'
    ]
    print(f"Using structured features: {structured_cols}")
    all_cols = pd.read_parquet(METADATA_PATH, columns=[]).columns
    for col in structured_cols:
       if col not in all_cols:
         print(f"Warning: The required structured column '{col}' was not found in the dataset.")
    
    full_dataset = PhishingDataset(
        metadata_path=METADATA_PATH,
        text_model_name=TEXT_MODEL_NAME,
        vision_model_name=VISION_MODEL_NAME,
        structured_cols=structured_cols
    )

    # Split dataset into training and validation sets
    train_size = int(TRAIN_VAL_SPLIT_RATIO * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # --- 2. Initialize Model, Loss, Optimizer ---
    model = PhishGuardModel(
        text_model_name=TEXT_MODEL_NAME,
        vision_model_name=VISION_MODEL_NAME,
        num_structured_features=len(structured_cols)
    ).to(device)

    criterion = nn.BCELoss() # Binary Cross-Entropy Loss for binary classification
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # --- 3. Training & Validation Loop ---
    best_val_accuracy = 0.0

    for epoch in range(EPOCHS):
        # -- Training Phase --
        model.train() # Set model to training mode
        total_train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Training]"):
            # Move batch to device
            text_inputs = {k: v.to(device) for k, v in batch['text_inputs'].items()}
            vision_inputs = {k: v.to(device) for k, v in batch['vision_inputs'].items()}
            structured_inputs = batch['structured_inputs'].to(device)
            labels = batch['label'].to(device).unsqueeze(1) # Add dimension for BCELoss

            # Forward pass
            outputs = model(text_inputs, vision_inputs, structured_inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        
        # -- Validation Phase --
        model.eval() # Set model to evaluation mode
        total_val_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad(): # No need to compute gradients
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Validation]"):
                text_inputs = {k: v.to(device) for k, v in batch['text_inputs'].items()}
                vision_inputs = {k: v.to(device) for k, v in batch['vision_inputs'].items()}
                structured_inputs = batch['structured_inputs'].to(device)
                labels = batch['label'].to(device).unsqueeze(1)

                outputs = model(text_inputs, vision_inputs, structured_inputs)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                # Store predictions and labels for metrics
                preds = (outputs > 0.5).float() # Convert probabilities to binary predictions (0 or 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')

        print(f"\nEpoch {epoch+1}/{EPOCHS}:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"  Val Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        
        # --- 4. Save the Best Model ---
        if accuracy > best_val_accuracy:
            best_val_accuracy = accuracy
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> New best model saved to {MODEL_SAVE_PATH} (Accuracy: {accuracy:.4f})")
            
    print("Training complete!")

if __name__ == '__main__':
    train_model()
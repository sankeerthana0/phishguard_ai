# src/dataset.py

import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoImageProcessor
from sklearn.preprocessing import StandardScaler
import joblib # To save the scaler

class PhishingDataset(Dataset):
    def __init__(self, metadata_path, text_model_name, vision_model_name, structured_cols, scaler=None):
        """
        Args:
            metadata_path (str): Path to the metadata.parquet file.
            text_model_name (str): Name of the Hugging Face text model.
            vision_model_name (str): Name of the Hugging Face vision model.
            structured_cols (list): List of column names for structured features.
            scaler (StandardScaler, optional): A pre-fitted scaler. If None, a new one will be fitted.
        """
        self.metadata = pd.read_parquet(metadata_path).reset_index(drop=True)
        self.structured_cols = structured_cols

        # Initialize processors and tokenizers
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.image_processor = AutoImageProcessor.from_pretrained(vision_model_name)
        
        # --- Handle structured data scaling ---
        if scaler:
            self.scaler = scaler
        else:
            # Fit a new scaler on the data and save it for later use in inference
            print("Fitting new scaler...")
            self.scaler = StandardScaler()
            self.metadata[self.structured_cols] = self.scaler.fit_transform(self.metadata[self.structured_cols])
            joblib.dump(self.scaler, 'models/scaler.pkl')
            print("Scaler saved to models/scaler.pkl")
            
        # Handle potential missing values after scaling (fill with 0, which is the mean)
        self.metadata[self.structured_cols] = self.metadata[self.structured_cols].fillna(0)


    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # 1. Get a single row of metadata
        row = self.metadata.iloc[idx]

        # 2. Process Text
        text = row.get('html_text', '') # Use .get for safety
        text_inputs = self.tokenizer(
            text,
            max_length=512,       # Standard max length for BERT-like models
            padding='max_length', # Pad to max_length
            truncation=True,      # Truncate if longer
            return_tensors="pt"   # Return PyTorch tensors
        )
        # Squeeze to remove the batch dimension (DataLoader will add it back)
        text_inputs = {key: val.squeeze(0) for key, val in text_inputs.items()}

        # 3. Process Vision (Screenshot)
        screenshot_path = row.get('screenshot_path')
        try:
            image = Image.open(screenshot_path).convert("RGB")
            vision_inputs = self.image_processor(image, return_tensors="pt")
            vision_inputs = {'pixel_values': vision_inputs['pixel_values'].squeeze(0)}
        except (FileNotFoundError, UnidentifiedImageError, TypeError):
            # If image is missing or corrupt, create a black tensor as a placeholder
            print(f"Warning: Could not load image {screenshot_path}. Using a black image placeholder.")
            # Get expected image dimensions from the processor
            img_size = self.image_processor.size
            black_image_tensor = torch.zeros(3, img_size['height'], img_size['width'])
            vision_inputs = {'pixel_values': black_image_tensor}


        # 4. Process Structured Data
        structured_data = row[self.structured_cols].values.astype('float32')
        structured_inputs = torch.tensor(structured_data, dtype=torch.float32)

        # 5. Get Label
        label = torch.tensor(row['label'], dtype=torch.float32)

        return {
            'text_inputs': text_inputs,
            'vision_inputs': vision_inputs,
            'structured_inputs': structured_inputs,
            'label': label
        }
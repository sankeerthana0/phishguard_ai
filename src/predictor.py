# src/predictor.py
import pandas as pd
import torch
import joblib
from PIL import Image
from transformers import AutoTokenizer, AutoImageProcessor

# Import your custom classes
from src.model import PhishGuardModel
from src.data_processor import process_url # We need this to process new URLs

class Predictor:
    def __init__(self, model_path, scaler_path, text_model_name, vision_model_name, structured_cols):
        """
        Initializes the predictor by loading the model and all necessary artifacts.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Predictor using device: {self.device}")

        # Load the trained model
        self.model = PhishGuardModel(
            text_model_name=text_model_name,
            vision_model_name=vision_model_name,
            num_structured_features=len(structured_cols)
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval() # Set to evaluation mode
        print("Model loaded successfully.")

        # Load the scaler
        self.scaler = joblib.load(scaler_path)
        print("Scaler loaded successfully.")
        self.structured_cols = structured_cols

        # Load tokenizers and image processors
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.image_processor = AutoImageProcessor.from_pretrained(vision_model_name)
        print("Tokenizer and Image Processor loaded.")

    def predict(self, url):
        """
        Takes a single URL string and returns the risk score and extracted features.
        """
        # 1. Extract features from the live URL
        # We'll save the screenshot to a temporary path
        temp_screenshot_path = "temp_screenshot.png"
        try:
            features = process_url(url, temp_screenshot_path)
            features['screenshot_path'] = temp_screenshot_path # Ensure path is correct
        except Exception as e:
            print(f"Error processing URL {url}: {e}")
            return 0.0, {"error": str(e)} # Return a safe score and error message

        # 2. Pre-process inputs for the model (mirroring the Dataset class)
        
        # --- Text ---
        text = features.get('html_text', '')
        text_inputs = self.tokenizer(text, max_length=512, padding='max_length', truncation=True, return_tensors="pt")
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

        # --- Vision ---
        try:
            image = Image.open(features['screenshot_path']).convert("RGB")
            vision_inputs = self.image_processor(image, return_tensors="pt")
            vision_inputs = {k: v.to(self.device) for k, v in vision_inputs.items()}
        except Exception:
            # Create a placeholder if the screenshot failed
            img_size = self.image_processor.size
            black_image_tensor = torch.zeros(1, 3, img_size['height'], img_size['width'], device=self.device)
            vision_inputs = {'pixel_values': black_image_tensor}

        # --- Structured ---
        # Create a DataFrame to ensure column order is correct for the scaler
        structured_df = pd.DataFrame([features])
        # Ensure all required columns exist, fill missing with 0
        for col in self.structured_cols:
            if col not in structured_df.columns:
                structured_df[col] = 0
        
        structured_data_scaled = self.scaler.transform(structured_df[self.structured_cols])
        structured_inputs = torch.tensor(structured_data_scaled, dtype=torch.float32).to(self.device)

        # 3. Run inference
        with torch.no_grad():
            prediction = self.model(text_inputs, vision_inputs, structured_inputs)
        
        risk_score = prediction.item()
        
        # Return the probability and the extracted features for display
        return risk_score, features
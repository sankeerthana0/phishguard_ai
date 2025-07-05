# app/app.py

import streamlit as st
import pandas as pd
import os
import sys
import time

# --- Path Setup ---
# This allows us to import from the 'src' directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.predictor import Predictor

# --- Page Configuration (MUST be the first Streamlit command) ---
st.set_page_config(
    page_title="PhishGuard AI | Advanced Phishing Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Constants ---
TEXT_MODEL_NAME = 'distilbert-base-uncased'
VISION_MODEL_NAME = 'google/vit-base-patch16-224-in21k'
MODEL_PATH = 'models/phishguard_model.pt'
SCALER_PATH = 'models/scaler.pkl'

# --- Load Model (Cached for performance) ---
@st.cache_resource
def load_system():
    """Loads the entire prediction system: model, scaler, and processors."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        st.error(f"Model ({MODEL_PATH}) or scaler ({SCALER_PATH}) not found. Please run `src/train.py` first.")
        return None
    
    structured_cols = ['url_len', 'hostname_len', 'path_len', 'num_dots', 'has_ip', 'domain_age']
    predictor = Predictor(
        model_path=MODEL_PATH,
        scaler_path=SCALER_PATH,
        text_model_name=TEXT_MODEL_NAME,
        vision_model_name=VISION_MODEL_NAME,
        structured_cols=structured_cols
    )
    return predictor

predictor = load_system()

# --- Helper Functions for UI ---
def display_risk_gauge(score):
    """Displays a custom gauge-like meter for the risk score."""
    if score > 0.7:
        color = "red"
        label = "High Risk"
    elif score > 0.4:
        color = "orange"
        label = "Suspicious"
    else:
        color = "green"
        label = "Likely Safe"
    
    st.markdown(f"""
    <div style="background-color: #262730; border-radius: 10px; padding: 20px;">
        <h3 style="color: white; text-align: center;">Risk Assessment</h3>
        <div style="text-align: center; font-size: 48px; font-weight: bold; color: {color};">
            {score*100:.1f}%
        </div>
        <p style="text-align: center; font-size: 24px; color: {color};">{label}</p>
    </div>
    """, unsafe_allow_html=True)

def display_feature_card(title, value, help_text=""):
    """Creates a visually distinct card for displaying a single feature."""
    st.markdown(f"""
    <div style="background-color: #262730; border-radius: 10px; padding: 15px; margin-bottom: 10px;">
        <h4 style="color: #fafafa; margin: 0; font-size: 16px;">{title}</h4>
        <p style="color: #00A9FF; font-size: 22px; font-weight: bold; margin: 5px 0 0 0;">{value}</p>
    </div>
    """, unsafe_allow_html=True)


# --- UI Layout ---

# --- Sidebar ---
with st.sidebar:
    st.image("https://raw.githubusercontent.com/streamlit/templates/main/multipage-app-template/st_logo_grayscale.png", width=100) # Placeholder logo
    st.title("PhishGuard AI")
    st.info(
        "This is an advanced, end-to-end phishing detection system. "
        "It leverages a multi-modal AI model that analyzes:\n"
        "- **URL patterns** (length, keywords, etc.)\n"
        "- **Website Content** (via Transformers)\n"
        "- **Visual Appearance** (via Vision Models)\n"
        "to provide a comprehensive risk score."
    )
    st.markdown("---")
    st.write("Made with ❤️ using Streamlit & PyTorch")

# --- Main Page ---
st.title("🛡️ Advanced Phishing Website Detector")
st.markdown("Enter a URL below to get a real-time analysis and risk assessment.")

# URL Input Form
with st.form(key="url_form"):
    url_input = st.text_input(
        "Enter URL", 
        "https://www.google.com", 
        label_visibility="collapsed",
        placeholder="e.g., https://www.google.com"
    )
    submit_button = st.form_submit_button(label="Analyze URL", use_container_width=True)


# --- Analysis Section ---
if submit_button:
    if predictor is None:
        st.error("System is not available. Please check model files.")
        st.stop()

    if not url_input or not url_input.startswith(('http://', 'https://')):
        st.warning("Please enter a full and valid URL (including http:// or https://).")
    else:
        # Show a placeholder while processing
        with st.spinner("Analyzing... This involves live scraping and running multiple AI models, please wait..."):
            start_time = time.time()
            risk_score, features = predictor.predict(url_input)
            end_time = time.time()
            processing_time = end_time - start_time

        st.markdown("---")
        st.header("Analysis Report")
        
        # --- Top Row: Gauge and Screenshot ---
        col1, col2 = st.columns([1, 2]) # Give more space to the screenshot
        
        with col1:
            display_risk_gauge(risk_score)
            st.metric("Analysis Time", f"{processing_time:.2f}s")
        
        with col2:
            st.subheader("Website Snapshot")
            if "error" in features or not os.path.exists(features.get('screenshot_path', '')):
                st.error("Could not capture a screenshot of the website. It might be down or blocking automated tools.")
            else:
                st.image(features['screenshot_path'], caption="Screenshot of the rendered webpage", use_container_width=True)

        st.markdown("---")
        
        # --- Second Row: Key Features Breakdown ---
        st.header("Key Feature Analysis")
        st.write("Our model analyzes dozens of features. Here are some of the most important ones for this URL.")
        
        feat_col1, feat_col2, feat_col3 = st.columns(3)
        
        with feat_col1:
            display_feature_card("URL Length", features.get('url_len', 'N/A'))
            display_feature_card("IP in Hostname?", "Yes" if features.get('has_ip') == 1 else "No")
        
        with feat_col2:
            display_feature_card("Hostname Length", features.get('hostname_len', 'N/A'))
            age = features.get('domain_age', -1)
            age_display = f"{age} days" if age != -1 else "Not Found"
            display_feature_card("Domain Age", age_display)

        with feat_col3:
            display_feature_card("Path Length", features.get('path_len', 'N/A'))
            display_feature_card("Number of Dots (.)", features.get('num_dots', 'N/A'))
        
        # --- Third Row: Raw Data (in an expander) ---
        with st.expander("🔬 Click to view all extracted raw data"):
            features_to_show = features.copy()
            # Truncate the long HTML text for better display
            features_to_show['html_text'] = features_to_show.get('html_text', '')[:500] + "..."
            st.json(features_to_show)
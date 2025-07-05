# generate_dataset.py
import os
import pandas as pd
import requests
from tqdm import tqdm
import random
from io import StringIO
from src.data_processor import process_url

NUM_PHISHING_URLS = 200 # Start small to test (e.g., 200)
NUM_BENIGN_URLS = 200   # Increase later for a better model
PHISHTANK_URL = "http://data.phishtank.com/data/online-valid.csv"
TRANCO_URL = "https://tranco-list.eu/top-1m.csv.zip"
PROCESSED_DATA_DIR = "data/processed"
SCREENSHOTS_DIR = os.path.join(PROCESSED_DATA_DIR, "screenshots")
METADATA_FILE = os.path.join(PROCESSED_DATA_DIR, "metadata.parquet")


def fetch_phishing_urls():
    """Downloads the latest verified phishing URLs from PhishTank."""
    print("Fetching phishing URLs from PhishTank...")
    try:
        # Define a browser-like User-Agent header
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Use the 'requests' library to make the request with the header
        response = requests.get(PHISHTANK_URL, headers=headers)
        
        # Check if the request was successful
        response.raise_for_status()  # This will raise an error if the status is not 200 (OK)
        
        # Use StringIO to let pandas read the text content of the response
        csv_data = StringIO(response.text)
        
        df = pd.read_csv(csv_data)
        urls = df['url'].tolist()
        print(f"  -> Found {len(urls)} phishing URLs.")
        return urls
    except Exception as e:
        print(f"Error fetching PhishTank data: {e}")
        return []

def fetch_benign_urls():
    print("Fetching benign URLs...")
    df = pd.read_csv(TRANCO_URL, compression='zip', header=None, names=['rank', 'domain'])
    return ["http://" + domain for domain in df['domain']]

def main():
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
    
    phishing_urls = fetch_phishing_urls()
    benign_urls = fetch_benign_urls()
    
    urls_to_process = (
        [(url, 1) for url in phishing_urls[:NUM_PHISHING_URLS]] +
        [(url, 0) for url in benign_urls[:NUM_BENIGN_URLS]]
    )
    random.shuffle(urls_to_process)
    
    all_features = []
    for i, (url, label) in enumerate(tqdm(urls_to_process, desc="Processing URLs")):
        screenshot_path = os.path.join(SCREENSHOTS_DIR, f"screenshot_{i}.png")
        try:
            features = process_url(url, screenshot_path)
            if features.get('screenshot_path') is None: continue
            features['url'], features['label'] = url, label
            all_features.append(features)
        except Exception as e:
            print(f"Error processing {url}: {e}. Skipping.")
    
    df = pd.DataFrame(all_features)
    df['domain_age'] = df['domain_age'].fillna(-1).astype(int)
    numeric_cols = ['url_len', 'hostname_len', 'path_len', 'num_dots', 'has_ip']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    df.dropna(subset=['screenshot_path', 'html_text'], inplace=True)
    df.to_parquet(METADATA_FILE)
    
    print(f"\nDataset generation complete. Metadata saved to: {METADATA_FILE}")
    print(f"Processed {len(df)} URLs. Now run 'python -m src.train'.")

if __name__ == "__main__":
    main()
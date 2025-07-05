import whois
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from datetime import datetime

def process_url(url: str, screenshot_path: str):
    features = {}
    try:
        parsed_url = urlparse(url)
        features['url_len'] = len(url)
        features['hostname_len'] = len(parsed_url.hostname or "")
        features['path_len'] = len(parsed_url.path or "")
        features['num_dots'] = url.count('.')
        features['has_ip'] = 1 if parsed_url.hostname and parsed_url.hostname.replace('.', '').isnumeric() else 0
    except Exception as e:
        print(f"URL parsing error for {url}: {e}")
        return {} # Return empty on fundamental failure
        
    try:
        w = whois.whois(parsed_url.hostname)
        if w.creation_date:
            creation_date = w.creation_date[0] if isinstance(w.creation_date, list) else w.creation_date
            features['domain_age'] = (datetime.now() - creation_date).days
        else:
            features['domain_age'] = -1
    except Exception:
        features['domain_age'] = -1

    try:
        response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)
        features['html_text'] = ' '.join(text.split())[:5000]
    except Exception:
        features['html_text'] = ""

    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1280,720")
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        driver.get(url)
        driver.save_screenshot(screenshot_path)
        driver.quit()
        features['screenshot_path'] = screenshot_path
    except Exception as e:
        print(f"Screenshot failed for {url}: {e}")
        features['screenshot_path'] = None
        
    return features
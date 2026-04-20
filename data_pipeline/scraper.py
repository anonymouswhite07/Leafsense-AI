import os
import time
import argparse
import logging
import requests
import base64
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

RAW_DATA_DIR = "../data/raw"
KEYWORDS = ["disease", "infection", "fungus", "spots", "blight", "leaf damage", "unhealthy leaves closeup", "crop disease field"]

class ImageScraper:
    def __init__(self, headless=True):
        self.headless = headless
        options = webdriver.ChromeOptions()
        if self.headless:
            options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        os.environ['WDM_LOG_LEVEL'] = '0'
        
        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=options
        )
        self.driver.set_page_load_timeout(30)

    def scrape_urls(self, url, selectors, limit):
        self.driver.get(url)
        image_urls = set()
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        no_change_count = 0
        
        while len(image_urls) < limit:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            
            # Extract
            thumbnails = []
            for sel in selectors:
                thumbnails.extend(self.driver.find_elements(By.CSS_SELECTOR, sel))
                
            for img in thumbnails:
                if len(image_urls) >= limit: break
                src = img.get_attribute('src')
                data_src = img.get_attribute('data-src')
                link = data_src if data_src else src
                if link and link not in image_urls:
                    image_urls.add(link)

            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                no_change_count += 1
                if no_change_count >= 3: 
                    break # Pagination Handling feature 3
            else:
                no_change_count = 0
            last_height = new_height
            
        return list(image_urls)
        
    def scrape_bing(self, query, limit):
        search_query = urllib.parse.quote_plus(query)
        url = f"https://www.bing.com/images/search?q={search_query}"
        return self.scrape_urls(url, ["img.mimg"], limit)
        
    def scrape_ddg(self, query, limit):
        search_query = urllib.parse.quote_plus(query)
        url = f"https://duckduckgo.com/?q={search_query}&iax=images&ia=images"
        return self.scrape_urls(url, ["img.tile--img__img", "img.js-lazyload"], limit)

    def close(self):
        self.driver.quit()

def expand_queries(base_query):
    # FEATURE 1: MULTI-QUERY EXPANSION
    parts = base_query.lower().replace("leaf", "").replace("disease", "").strip().split()
    crop = parts[0] if len(parts) > 0 else "plant"
    condition = " ".join(parts[1:]) if len(parts) > 1 else ""
    
    queries = [base_query]
    for kw in KEYWORDS:
        queries.append(f"{crop} {kw} {condition}".strip())
    
    # Deduplicate permutations
    return list(set(queries))
    
def download_image(url, save_path):
    try:
        if url.startswith("data:image"):
            head, data = url.split(',', 1)
            img_data = base64.b64decode(data)
            img = Image.open(BytesIO(img_data)).convert('RGB')
            img.save(save_path)
            return True
        else:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content)).convert('RGB')
                img.save(save_path)
                return True
    except Exception as e:
        return False
    return False

def process_base_query(base_query, total_limit=1000):
    query_folder = os.path.join(RAW_DATA_DIR, base_query.replace(" ", "_"))
    os.makedirs(query_folder, exist_ok=True)
    
    variations = expand_queries(base_query)
    logging.info(f"Expanded '{base_query}' into {len(variations)} variations.")
    
    urls_to_download = set()
    limit_per_variation = max(50, total_limit // len(variations))
    
    scraper = ImageScraper(headless=True)
    try:
        for var in variations:
            if len(urls_to_download) >= total_limit: break
            logging.info(f"Scraping variation: {var}")
            bing_urls = scraper.scrape_bing(var, limit_per_variation)
            ddg_urls = scraper.scrape_ddg(var, limit_per_variation)
            
            for u in bing_urls + ddg_urls:
                if len(urls_to_download) < total_limit:
                    urls_to_download.add(u)
    finally:
        scraper.close()
        
    logging.info(f"Aggregated {len(urls_to_download)} unique URLs overall. Downloading...")
    
    success = 0
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for i, url in enumerate(urls_to_download):
            path = os.path.join(query_folder, f"scraped_{i:06d}.jpg")
            if not os.path.exists(path):
                futures.append(executor.submit(download_image, url, path))
            
        for future in tqdm(futures, desc=f"Downloading {base_query}"):
            if future.result():
                success += 1
                
    logging.info(f"✅ Successfully persisted {success} images for class '{base_query}'.")

def main():
    parser = argparse.ArgumentParser(description="Multi-Source Image Scraper Pipeline")
    parser.add_argument("--query", type=str, required=True, help="Comma-separated base queries")
    parser.add_argument("--limit", type=int, default=1000, help="Target total images per base query")
    args = parser.parse_args()

    base_queries = [q.strip() for q in args.query.split(",") if q.strip()]
    
    for bq in base_queries:
        process_base_query(bq, args.limit)

if __name__ == "__main__":
    main()

import sys
sys.path.append('data_pipeline')
from scraper import process_base_query

queries = [
    "Apple Scab",
    "Apple healthy",
    "Corn Rust",
    "Corn healthy",
    "Tomato healthy",
    "Healthy Leaf"
]

for q in queries:
    print(f"--- Scraping {q} ---")
    try:
        process_base_query(q, total_limit=50)
    except Exception as e:
        print(f"Error scraping {q}: {e}")

print("Scraping finished.")

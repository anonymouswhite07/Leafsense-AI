import os
import io
import shutil
import json
import random
import logging
from PIL import Image
import imagehash
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')

RAW_DATA_DIR = "../data/raw"
OUTPUT_DIR = "../data"
LABELS_JSON = "../model/class_labels.json"
REPORT_JSON = "../data/dataset_report.json"

IMG_SIZE = 224
MIN_SIZE = 100
HASH_CUTOFF = 5 # max phash difference to be considered a duplicate

def clean_and_deduplicate(raw_subdir):
    """ Cleans corrupted/small images, normalizes format, and removes duplicates """
    clean_images = []
    seen_hashes = {}
    duplicates = 0
    corrupted = 0
    small_size = 0
    
    files = [f for f in os.listdir(raw_subdir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    for filename in tqdm(files, desc=f"Cleaning {os.path.basename(raw_subdir)}", leave=False):
        filepath = os.path.join(raw_subdir, filename)
        try:
            with Image.open(filepath) as img:
                # 1. Validation
                img.verify()
            
            with Image.open(filepath) as img:
                if img.width < MIN_SIZE or img.height < MIN_SIZE:
                    small_size += 1
                    continue
                
                # 2. Conversion
                img = img.convert("RGB")
                img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
                
                # 3. Deduplication via Hash
                img_hash = str(imagehash.phash(img))
                
                is_duplicate = False
                for existing_hash in seen_hashes.keys():
                    if imagehash.hex_to_hash(img_hash) - imagehash.hex_to_hash(existing_hash) < HASH_CUTOFF:
                        is_duplicate = True
                        break
                
                if is_duplicate:
                    duplicates += 1
                else:
                    seen_hashes[img_hash] = True
                    clean_images.append((filepath, img))
                    
        except Exception as e:
            corrupted += 1
            
    return clean_images, {"duplicates": duplicates, "corrupted": corrupted, "small": small_size}

def create_structured_splits(queries_data_map):
    """
    Splits data into train/val/test and saves to disk.
    Auto-labels queries directly based on CamelCase transformation.
    """
    random.seed(42)
    splits = {"train": 0.8, "val": 0.1, "test": 0.1}
    
    for split in splits.keys():
        os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)
        
    labels_dict = {}
    report = {"total_processed": 0, "classes": {}, "removed_stats": {}}
    
    for idx, (query_name, clean_payload) in enumerate(queries_data_map.items()):
        images, stats = clean_payload
        report["removed_stats"][query_name] = stats
        
        # Phase 3: Auto-labeling mapping ("tomato leaf blight" -> "Tomato___Leaf_Blight")
        parts = [p.capitalize() for p in query_name.replace("_", " ").split(" ")]
        plant = parts[0]
        disease = "_".join(parts[1:]) if len(parts) > 1 else "Healthy"
        clean_label = f"{plant}___{disease}"
        
        labels_dict[str(idx)] = clean_label
        report["classes"][clean_label] = len(images)
        report["total_processed"] += len(images)
        
        # Phase 4 Structuring
        random.shuffle(images)
        train_idx = int(len(images) * splits["train"])
        val_idx = train_idx + int(len(images) * splits["val"])
        
        partitions = {
            "train": images[:train_idx],
            "val": images[train_idx:val_idx],
            "test": images[val_idx:]
        }
        
        for p_name, p_images in partitions.items():
            class_dir = os.path.join(OUTPUT_DIR, p_name, clean_label)
            os.makedirs(class_dir, exist_ok=True)
            for i, (old_path, img_obj) in enumerate(p_images):
                save_path = os.path.join(class_dir, f"img_{i:05d}.jpg")
                img_obj.save(save_path, "JPEG", quality=90)
                
    # Save Labels
    with open(LABELS_JSON, "w") as f:
        json.dump(labels_dict, f, indent=4)
        
    # Save Report
    with open(REPORT_JSON, "w") as f:
        json.dump(report, f, indent=4)
        
    logging.info("Dataset Structure Completed Successfully.")
    logging.info(f"Report: {json.dumps(report, indent=2)}")

def main():
    if not os.path.exists(RAW_DATA_DIR):
        logging.error(f"Raw data directory {RAW_DATA_DIR} does not exist.")
        return
        
    directories = [d for d in os.listdir(RAW_DATA_DIR) if os.path.isdir(os.path.join(RAW_DATA_DIR, d))]
    
    queries_data_map = {}
    for d in directories:
        target = os.path.join(RAW_DATA_DIR, d)
        logging.info(f"Processing Raw Directory: {d}")
        clean_imgs, stats = clean_and_deduplicate(target)
        queries_data_map[d] = (clean_imgs, stats)
        
    create_structured_splits(queries_data_map)

if __name__ == "__main__":
    main()

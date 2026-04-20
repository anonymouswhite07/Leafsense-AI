import os
import shutil
import json
from PIL import Image

REVIEW_DIR = "../data/review"
TRAIN_DIR = "../data/train"
LABELS_PATH = "../model/class_labels.json"

def main():
    if not os.path.exists(REVIEW_DIR) or len(os.listdir(REVIEW_DIR)) == 0:
        print("✅ No images in the review queue. Dataset is clean.")
        return
        
    with open(LABELS_PATH, "r") as f:
        labels = json.load(f)
        
    print("\n🌿 LeafSense Human-in-the-Loop Review Tool 🌿\n")
    print(f"Available Labels:")
    for i, lbl in enumerate(labels):
        print(f"[{i}] {lbl}")
    print("[d] DELETE Image")
    print("[s] SKIP Image\n")
    
    files = os.listdir(REVIEW_DIR)
    for file in files:
        # File syntax from pipeline: ACTUAL___pred_PREDICTED___CONF___filename.jpg
        parts = file.split("___")
        if len(parts) >= 4:
            actual = parts[0]
            pred = parts[1].replace("pred_", "")
            conf = parts[2]
            
            print(f"\nReviewing file: {file}")
            print(f"Assigned Label: {actual}")
            print(f"AI Suspects: {pred} (Confidence: {conf})")
            
            filepath = os.path.join(REVIEW_DIR, file)
            img = Image.open(filepath)
            img.show() # Pops up default OS image viewer
            
            choice = input(f"Provide correct class ID [0-{len(labels)-1}], 'd' to delete, or 's' to skip: ").strip().lower()
            
            if choice == 'd':
                os.remove(filepath)
                print(f"Deleted {file}")
            elif choice == 's':
                print("Skipped.")
            elif choice.isdigit() and int(choice) < len(labels):
                corrected_lbl = labels[int(choice)]
                dest_dir = os.path.join(TRAIN_DIR, corrected_lbl)
                os.makedirs(dest_dir, exist_ok=True)
                
                # Cleanup filename for clean structured storage
                clean_filename = "___".join(parts[3:])
                dest_path = os.path.join(dest_dir, f"corrected_{clean_filename}")
                
                shutil.move(filepath, dest_path)
                print(f"✅ Moved to {corrected_lbl}")
            else:
                print("Invalid input. Skipped.")
                
    print("\nReview queue finished.")

if __name__ == "__main__":
    main()

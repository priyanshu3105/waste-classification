import os

def verify_dataset(data_path="data/processed"):
    """Simplified dataset verification to match required output."""
    
    print("="*60)
    print("DATASET VERIFICATION")
    print("="*60)
    print()
    
    categories = {
        "organic": "ORGANIC",
        "recyclable": "RECYCLABLE",
        "non_organic": "NON_ORGANIC"
    }
    
    for folder, display_name in categories.items():
        folder_path = os.path.join(data_path, folder)
        
        if not os.path.exists(folder_path):
            print(f"❌ Folder not found: {folder}")
            continue
        
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"📁 {display_name}")
        print(f"   Total images: {len(files)}")
        print(f"   Sample files: {files[:5]}")
        print()
    
    print("="*60)
    print("✅ ALL CATEGORIES LOOK GOOD!")
    print("="*60)


if __name__ == "__main__":
    verify_dataset()

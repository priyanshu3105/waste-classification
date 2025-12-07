

import os
import shutil
from pathlib import Path
import json
from collections import defaultdict
import random

# Correct mapping for your dataset
CLASS_MAPPING = {
    'organic': ['biological'],
    'recyclable': [
        'paper', 'cardboard', 'metal', 'plastic',
        'white-glass', 'green-glass', 'brown-glass'
    ],
    'non_organic': ['trash', 'shoes', 'clothes', 'battery']
}

TARGET_CLASSES = ['organic', 'recyclable', 'non_organic']


def map_to_target_class(original_class):
    """Map original dataset class to one of 3 target categories"""
    original = original_class.lower()

    for target, keywords in CLASS_MAPPING.items():
        for key in keywords:
            if key in original:
                return target
    return "non_organic"  # fallback


def ensure_clean_dir(path):
    """Delete folder if exists and recreate"""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def oversample_class(class_dir, target_count):
    """Duplicates images until class count reaches target_count"""
    files = os.listdir(class_dir)
    image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if len(image_files) == 0:
        return

    while len(image_files) < target_count:
        choice = random.choice(image_files)
        src = os.path.join(class_dir, choice)
        new_name = f"aug_{len(image_files)}_{choice}"
        dst = os.path.join(class_dir, new_name)
        shutil.copy2(src, dst)
        image_files.append(new_name)

    print(f"✓ Oversampled {class_dir} to {len(image_files)} images")


def process_dataset(raw_path, processed_path):
    """Main dataset processing with balancing"""

    # Detect dataset folder
    raw_classes = os.listdir(raw_path)
    raw_classes = [c for c in raw_classes if os.path.isdir(os.path.join(raw_path, c))]

    if len(raw_classes) == 0:
        print("❌ No dataset found in raw path.")
        return False

    print("Found raw classes:")
    for cls in raw_classes:
        print(f" - {cls}")

    # Prepare processed folder
    for cls in TARGET_CLASSES:
        ensure_clean_dir(os.path.join(processed_path, cls))

    stats = defaultdict(lambda: defaultdict(int))
    total_images = 0

    print("\n====================================")
    print("PROCESSING & MERGING CLASSES")
    print("====================================")

    # Step 1: Copy images to merged folders
    for original_class in raw_classes:
        src_dir = os.path.join(raw_path, original_class)
        target_class = map_to_target_class(original_class)
        dst_dir = os.path.join(processed_path, target_class)

        print(f"{original_class} → {target_class}")

        for img_file in os.listdir(src_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                src = os.path.join(src_dir, img_file)
                new_name = f"{target_class}_{original_class}_{img_file}"
                dst = os.path.join(dst_dir, new_name)
                shutil.copy2(src, dst)

                stats[target_class][original_class] += 1
                total_images += 1

    # Step 2: Count and oversample
    print("\n====================================")
    print("BALANCING CLASSES (OVERSAMPLING)")
    print("====================================")

    class_sizes = {
        cls: sum(stats[cls].values())
        for cls in TARGET_CLASSES
    }

    print("\nOriginal distribution:")
    print(json.dumps(class_sizes, indent=2))

    max_count = max(class_sizes.values())

    for cls in TARGET_CLASSES:
        class_dir = os.path.join(processed_path, cls)
        oversample_class(class_dir, max_count)

    # Step 3: Verify
    final_counts = {}
    for cls in TARGET_CLASSES:
        dir_path = os.path.join(processed_path, cls)
        final_counts[cls] = len([
            f for f in os.listdir(dir_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

    print("\n====================================")
    print("FINAL BALANCED CLASS DISTRIBUTION")
    print("====================================")

    print(json.dumps(final_counts, indent=2))

    # Step 4: Save mapping info
    mapping_info = {
        'class_mapping': CLASS_MAPPING,
        'original_counts': class_sizes,
        'balanced_counts': final_counts,
        'total_original_images': total_images
    }

    with open(os.path.join(processed_path, "mapping_info.json"), "w") as f:
        json.dump(mapping_info, f, indent=2)

    print(f"\n✓ Processed dataset saved to: {processed_path}")
    print(f"✓ Mapping info saved.")

    return True


if __name__ == "__main__":
    RAW_DATA_PATH = "data/garbage_classification"
    PROCESSED_DATA_PATH = "data/processed"

    print("====================================")
    print(" WASTE CLASSIFICATION DATA PROCESSING")
    print("====================================")

    success = process_dataset(RAW_DATA_PATH, PROCESSED_DATA_PATH)

    if success:
        print("\n✓ DATA PROCESSING SUCCESSFUL")
        print("Next step: Train your ResNet-50 model")
    else:
        print("\n❌ DATA PROCESSING FAILED")

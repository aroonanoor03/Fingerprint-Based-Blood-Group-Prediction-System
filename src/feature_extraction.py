import numpy as np
from skimage.feature import hog
from src.preprocessing import preprocess_image

def extract_hog_features(image, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2)):
    features = hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell,cells_per_block=cells_per_block,visualize=False)
    return features

def extract_dataset_features(image_records, limit=None):
    X = []
    y = []
    
    records_to_process = image_records[:limit] if limit is not None else image_records
    total = len(records_to_process)
    
    print(f"Extracting features for {total} images...")
    for i, record in enumerate(records_to_process):
        img_path = record["path"]
        label = record["blood_group"]
        
        # Preprocess using preprocessing module (Step 2)
        img_preprocessed = preprocess_image(img_path)
        if img_preprocessed is not None:
            # Extract HOG features (Step 3)
            features = extract_hog_features(img_preprocessed)
            X.append(features)
            y.append(label)
        
        # Simple progress update every 500 images
        if (i + 1) % 500 == 0 or (i + 1) == total:
            print(f"   Processed {i + 1}/{total} images...")
            
    return np.array(X), np.array(y)

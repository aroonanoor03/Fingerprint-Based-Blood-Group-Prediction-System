import cv2
import numpy as np

def preprocess_image(image_path, target_size=(224, 224)):
    # 1. Load the image
    img = cv2.imread(image_path)
    if img is None:
        return None

    # 2. Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Resize image to 224x224
    img_resized = cv2.resize(img_gray, target_size, interpolation=cv2.INTER_AREA)

    # 4. Apply median filter for noise removal (kernel size 3 is standard)
    img_filtered = cv2.medianBlur(img_resized, 3)

    # 5. Apply histogram equalization for enhancement
    img_enhanced = cv2.equalizeHist(img_filtered)

    # 6. Normalize pixel values to [0, 1]
    img_normalized = img_enhanced.astype(np.float32) / 255.0

    return img_normalized

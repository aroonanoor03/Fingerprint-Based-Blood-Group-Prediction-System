import os
import sys
import joblib
from src.preprocessing import preprocess_image
from src.feature_extraction import extract_hog_features

def predict_fingerprint_blood_group(image_path, model_path="models/svm_model.pkl"):
    """
    Loads the best trained model (Linear SVM), preprocesses a new fingerprint image,
    extracts HOG features, and predicts the associated blood group.
    
    Parameters:
        image_path (str): Path to the target fingerprint image.
        model_path (str): Path to the saved classifier file.
        
    Returns:
        str: Predicted blood group string (e.g. 'A+'), or None if failure.
    """
    # 1. Verify existence of model and image files
    if not os.path.exists(image_path):
        print(f"Error: Fingerprint image not found at '{image_path}'.")
        return None
        
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'. Please train the model first.")
        return None

    # 2. Load the best trained model
    print(f"Loading trained classifier from: {model_path}...")
    model = joblib.load(model_path)

    # 3. Apply preprocessing (Step 2)
    print("Preprocessing fingerprint image...")
    preprocessed = preprocess_image(image_path)
    if preprocessed is None:
        print("Error: Image preprocessing failed.")
        return None

    # 4. Extract HOG features (Step 3)
    print("Extracting HOG features...")
    features = extract_hog_features(preprocessed)
    
    # Reshape 1D feature vector to 2D array (1 sample, N features) as expected by scikit-learn
    features_2d = features.reshape(1, -1)

    # 5. Predict blood group class
    print("Running model inference...")
    prediction = model.predict(features_2d)
    predicted_blood_group = prediction[0]

    return predicted_blood_group

def main():
    # Allow passing image path as command line argument
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Default test image from A+ class
        image_path = "dataset_blood_group/A+/cluster_0_1001.BMP"
        print(f"No image path specified. Using default test image: {image_path}\n")

    predicted_bg = predict_fingerprint_blood_group(image_path)

    if predicted_bg is not None:
        print("\n" + "="*45)
        print(f" PREDICTED BLOOD GROUP: {predicted_bg}")
        print("="*45 + "\n")

if __name__ == "__main__":
    main()

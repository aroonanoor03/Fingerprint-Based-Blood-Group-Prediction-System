import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from src.feature_extraction import extract_dataset_features
from src.evaluation import evaluate_model

def load_dataset_records(dataset_dir="dataset_blood_group"):
    blood_groups = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]
    image_records = []
    
    for bg in blood_groups:
        bg_folder = os.path.join(dataset_dir, bg)
        if os.path.exists(bg_folder) and os.path.isdir(bg_folder):
            files = [f for f in os.listdir(bg_folder) if f.lower().endswith(".bmp")]
            for filename in files:
                image_records.append({
                    "path": os.path.join(bg_folder, filename),
                    "blood_group": bg
                })
    return image_records

def main():
    dataset_dir = "dataset_blood_group"
    models_dir = "models"

    print("=== Training Support Vector Machine (SVM) Classifier ===")
    
    # 1. Load the dataset file lists
    image_records = load_dataset_records(dataset_dir)
    if not image_records:
        print(f"Error: No images found in dataset directory '{dataset_dir}'.")
        return

    # 2. Extract HOG features and get labels
    X, y = extract_dataset_features(image_records, limit=None)
    
    # 3. Train-Test Split (80% training, 20% validation)
    print("Splitting dataset into train (80%) and test (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train samples: {X_train.shape[0]}")
    print(f"   Test samples:  {X_test.shape[0]}")

    # 4. Train the SVM Model
    print("Training Support Vector Machine (SVC) with linear kernel...")
    svm_clf = SVC(kernel="linear", C=1.0, random_state=42)
    svm_clf.fit(X_train, y_train)
    print("SVM model training completed!")

    # 5. Evaluate the model
    print("Evaluating SVM model on test set...")
    y_pred = svm_clf.predict(X_test)
    evaluate_model(y_test, y_pred, model_name="SVM")

    # 6. Save the trained model to disk
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "svm_model.pkl")
    joblib.dump(svm_clf, model_path)
    print(f"Successfully saved SVM model to: {model_path}")

if __name__ == "__main__":
    main()

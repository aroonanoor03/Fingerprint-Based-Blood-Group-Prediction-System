import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

def evaluate_model(y_true, y_pred, model_name="Model", results_dir="results"):
    """
    Calculates and prints model evaluation metrics (Accuracy, Precision, Recall).
    Generates and saves the confusion matrix plot.
    
    Parameters:
        y_true (numpy.ndarray): True labels.
        y_pred (numpy.ndarray): Predicted labels.
        model_name (str): Name of the model (used in outputs).
        results_dir (str): Directory where to save the confusion matrix plot.
        
    Returns:
        dict: A dictionary containing the calculated metrics.
    """
    # 1. Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # 2. Calculate precision and recall (using 'macro' averaging for multiclass classification)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\n=================== {model_name} Model Evaluation ===================")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} (Macro Average)")
    print(f"Recall:    {recall:.4f} (Macro Average)")
    print("===================================================================\n")

    # 3. Compute and plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    classes = sorted(list(set(y_true)))
    
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
    plt.title(f"Confusion Matrix - {model_name} Classifier", fontsize=14, fontweight="bold")
    plt.tight_layout()

    # Create results folder if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Save the plot
    plot_filename = f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
    plot_path = os.path.join(results_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Saved confusion matrix plot to: {plot_path}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm
    }

def compare_models(dataset_dir="dataset_blood_group", models_dir="models"):
    """
    Loads all trained models (SVM, Random Forest, CNN) and evaluates them on 
    the same test split to generate a comprehensive comparison table.
    """
    import joblib
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision.models import vgg16
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    
    # Import loading/extraction helpers locally to avoid cyclic imports
    from src.svm_model import load_dataset_records
    from src.feature_extraction import extract_dataset_features
    from src.cnn_model import preload_and_preprocess_all, FingerprintDataset

    print("=== Loading Dataset & Models for Evaluation & Comparison ===")
    
    image_records = load_dataset_records(dataset_dir)
    if not image_records:
        print(f"Error: No images found in dataset directory '{dataset_dir}'.")
        return

    # --- 1. Evaluate HOG-based models (SVM and Random Forest) ---
    print("\n[1/3] Evaluating HOG-based models (SVM and Random Forest)...")
    X_hog, y_hog = extract_dataset_features(image_records)
    
    # Split using same random_state to ensure identical test set partition
    _, X_test_hog, _, y_test_hog = train_test_split(
        X_hog, y_hog, test_size=0.2, random_state=42, stratify=y_hog
    )

    # A. Linear SVM
    svm_path = os.path.join(models_dir, "svm_model.pkl")
    if os.path.exists(svm_path):
        svm_model = joblib.load(svm_path)
        y_pred_svm = svm_model.predict(X_test_hog)
        acc_svm = accuracy_score(y_test_hog, y_pred_svm)
        prec_svm = precision_score(y_test_hog, y_pred_svm, average="macro", zero_division=0)
        rec_svm = recall_score(y_test_hog, y_pred_svm, average="macro", zero_division=0)
        print("   SVM evaluation completed.")
    else:
        print("   Warning: SVM model file not found.")
        acc_svm, prec_svm, rec_svm = 0.0, 0.0, 0.0

    # B. Random Forest
    rf_path = os.path.join(models_dir, "random_forest_model.pkl")
    if os.path.exists(rf_path):
        rf_model = joblib.load(rf_path)
        y_pred_rf = rf_model.predict(X_test_hog)
        acc_rf = accuracy_score(y_test_hog, y_pred_rf)
        prec_rf = precision_score(y_test_hog, y_pred_rf, average="macro", zero_division=0)
        rec_rf = recall_score(y_test_hog, y_pred_rf, average="macro", zero_division=0)
        print("   Random Forest evaluation completed.")
    else:
        print("   Warning: Random Forest model file not found.")
        acc_rf, prec_rf, rec_rf = 0.0, 0.0, 0.0

    # --- 2. Evaluate CNN model (VGG16) ---
    print("\n[2/3] Evaluating VGG16 CNN model...")
    cnn_path = os.path.join(models_dir, "cnn_model.pth")
    if os.path.exists(cnn_path):
        # Preload raw images and labels
        images, labels = preload_and_preprocess_all(image_records)
        indices = np.arange(len(images))
        
        # Test split for CNN (using identical partition split logic)
        _, test_idx = train_test_split(
            indices, test_size=0.2, random_state=42, stratify=labels.numpy()
        )
        
        test_dataset = FingerprintDataset(images[test_idx], labels[test_idx])
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        # Rebuild VGG16 structure
        model = vgg16()
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 8)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(cnn_path, map_location=device))
        model.to(device)
        model.eval()

        all_preds = []
        all_targets = []
        blood_groups = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]

        with torch.no_grad():
            for batch_imgs, batch_lbls in test_loader:
                batch_imgs = batch_imgs.to(device)
                outputs = model(batch_imgs)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(batch_lbls.numpy())

        # Map targets and predictions to blood group strings
        y_test_cnn = np.array([blood_groups[idx] for idx in all_targets])
        y_pred_cnn = np.array([blood_groups[idx] for idx in all_preds])

        acc_cnn = accuracy_score(y_test_cnn, y_pred_cnn)
        prec_cnn = precision_score(y_test_cnn, y_pred_cnn, average="macro", zero_division=0)
        rec_cnn = recall_score(y_test_cnn, y_pred_cnn, average="macro", zero_division=0)
        print("   CNN evaluation completed.")
    else:
        print("   Warning: CNN model weights file not found.")
        acc_cnn, prec_cnn, rec_cnn = 0.0, 0.0, 0.0

    # --- 3. Comparison Results Display ---
    print("\n" + "="*70)
    print("                      MODEL COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Model Name':<20} | {'Accuracy':<10} | {'Precision (Macro)':<18} | {'Recall (Macro)':<15}")
    print("-"*70)
    if acc_svm > 0:
        print(f"{'Linear SVM':<20} | {acc_svm:<10.4f} | {prec_svm:<18.4f} | {rec_svm:<15.4f}")
    if acc_rf > 0:
        print(f"{'Random Forest':<20} | {acc_rf:<10.4f} | {prec_rf:<18.4f} | {rec_rf:<15.4f}")
    if acc_cnn > 0:
        print(f"{'VGG16 CNN':<20} | {acc_cnn:<10.4f} | {prec_cnn:<18.4f} | {rec_cnn:<15.4f}")
    print("="*70)

    # Determine best model
    results = {}
    if acc_svm > 0: results["Linear SVM"] = acc_svm
    if acc_rf > 0: results["Random Forest"] = acc_rf
    if acc_cnn > 0: results["VGG16 CNN"] = acc_cnn

    if results:
        best_model = max(results, key=results.get)
        print(f"\n>>> Best Performing Model: {best_model} ({results[best_model]*100:.2f}% Accuracy)")
        print("="*70 + "\n")
    else:
        print("No models were successfully evaluated.")
        best_model = "None"

    # --- 4. Save results to files for supervisor ---
    os.makedirs("results", exist_ok=True)
    
    # Save as Markdown
    md_path = os.path.join("results", "model_comparison.md")
    with open(md_path, "w") as f:
        f.write("# Fingerprint Blood Group Prediction - Model Comparison Report\n\n")
        f.write("This report summarizes the evaluation metrics of the trained classifiers evaluated on the stratified test set partition.\n\n")
        f.write("## Evaluation Metrics Summary Table\n\n")
        f.write("| Model Name | Accuracy | Precision (Macro) | Recall (Macro) |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        if acc_svm > 0:
            f.write(f"| Linear SVM | {acc_svm:.4f} ({acc_svm*100:.2f}%) | {prec_svm:.4f} | {rec_svm:.4f} |\n")
        if acc_rf > 0:
            f.write(f"| Random Forest | {acc_rf:.4f} ({acc_rf*100:.2f}%) | {prec_rf:.4f} | {rec_rf:.4f} |\n")
        if acc_cnn > 0:
            f.write(f"| VGG16 CNN | {acc_cnn:.4f} ({acc_cnn*100:.2f}%) | {prec_cnn:.4f} | {rec_cnn:.4f} |\n")
        f.write("\n")
        if results:
            f.write(f"**Best Performing Model**: **{best_model} ({results[best_model]*100:.2f}% Accuracy)**\n")
            
    # Save as Text
    txt_path = os.path.join("results", "model_comparison.txt")
    with open(txt_path, "w") as f:
        f.write("="*70 + "\n")
        f.write("                      MODEL COMPARISON SUMMARY\n")
        f.write("="*70 + "\n")
        f.write(f"{'Model Name':<20} | {'Accuracy':<10} | {'Precision (Macro)':<18} | {'Recall (Macro)':<15}\n")
        f.write("-"*70 + "\n")
        if acc_svm > 0:
            f.write(f"{'Linear SVM':<20} | {acc_svm:<10.4f} | {prec_svm:<18.4f} | {rec_svm:<15.4f}\n")
        if acc_rf > 0:
            f.write(f"{'Random Forest':<20} | {acc_rf:<10.4f} | {prec_rf:<18.4f} | {rec_rf:<15.4f}\n")
        if acc_cnn > 0:
            f.write(f"{'VGG16 CNN':<20} | {acc_cnn:<10.4f} | {prec_cnn:<18.4f} | {rec_cnn:<15.4f}\n")
        f.write("="*70 + "\n")
        if results:
            f.write(f"\n>>> Best Performing Model: {best_model} ({results[best_model]*100:.2f}% Accuracy)\n")
            f.write("="*70 + "\n")
            
    print(f"Saved comparison reports to:\n   - {md_path}\n   - {txt_path}")

if __name__ == "__main__":
    compare_models()

import os
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

def main():
    # Define directories
    dataset_dir = "dataset_blood_group"
    results_dir = "results"
    models_dir = "models"

    # Ensure results and models folders exist
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print("=== Fingerprint Based Blood Group Prediction System ===")
    print("Step 1: Dataset Loading & Exploratory Data Analysis (EDA)\n")

    # Define the 8 blood groups we expect
    blood_groups = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]

    # Check if dataset directory exists
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory '{dataset_dir}' not found.")
        return

    # Load file paths and blood group labels
    image_records = []
    for bg in blood_groups:
        bg_folder = os.path.join(dataset_dir, bg)
        if os.path.exists(bg_folder) and os.path.isdir(bg_folder):
            # Find all BMP images in this blood group's folder
            files = [f for f in os.listdir(bg_folder) if f.lower().endswith(".bmp")]
            for filename in files:
                filepath = os.path.join(bg_folder, filename)
                image_records.append({
                    "path": filepath,
                    "blood_group": bg
                })
        else:
            print(f"Warning: Blood group folder '{bg}' not found in '{dataset_dir}'.")

    # Create a Pandas DataFrame for easy analysis
    df = pd.DataFrame(image_records)

    if len(df) == 0:
        print("No images found in the dataset folder.")
        return

    # 1. Count total images
    total_images = len(df)
    print(f"1. Total images found: {total_images}")

    # 2. Show blood group distribution
    print("\n2. Blood Group Distribution:")
    distribution = df["blood_group"].value_counts()
    print(distribution)

    # Save blood group distribution plot
    plt.figure(figsize=(8, 5))
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f1c40f", "#9b59b6", "#e67e22", "#1abc9c", "#95a5a6"]
    distribution.plot(kind="bar", color=colors, edgecolor="black")
    plt.title("Fingerprint Images Count by Blood Group Class", fontsize=14)
    plt.xlabel("Blood Group", fontsize=12)
    plt.ylabel("Number of Images", fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    
    dist_plot_path = os.path.join(results_dir, "blood_group_distribution.png")
    plt.savefig(dist_plot_path)
    plt.close()
    print(f"   Saved distribution plot to: {dist_plot_path}")

    # 3. Check image sizes
    print("\n3. Checking image sizes...")
    sizes = []
    for path in df["path"]:
        with Image.open(path) as img:
            sizes.append(img.size)
    
    df["size"] = sizes
    size_distribution = df["size"].value_counts()
    print("Image Sizes Distribution:")
    for size, count in size_distribution.items():
        print(f"   Size (Width, Height) = {size}: {count} images")

    # 4. Display sample fingerprint images
    print("\n4. Selecting sample images...")
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()

    for idx, bg in enumerate(blood_groups):
        # Find images belonging to this blood group class
        bg_subset = df[df["blood_group"] == bg]
        if len(bg_subset) > 0:
            # Select the first image in the list
            sample_row = bg_subset.iloc[0]
            img_path = sample_row["path"]
            img_size = sample_row["size"]
            
            with Image.open(img_path) as img:
                axes[idx].imshow(img, cmap="gray")
                axes[idx].set_title(f"Group: {bg}\nSize: {img_size}", fontsize=10)
                axes[idx].axis("off")
        else:
            axes[idx].text(0.5, 0.5, f"No data for {bg}", ha="center", va="center")
            axes[idx].axis("off")

    plt.suptitle("Sample Fingerprints for Each Blood Group", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    samples_plot_path = os.path.join(results_dir, "sample_fingerprints.png")
    plt.savefig(samples_plot_path)
    plt.close()
    print(f"   Saved sample fingerprint grid to: {samples_plot_path}")

    print("\n=== EDA Step Completed Successfully ===")

if __name__ == "__main__":
    main()

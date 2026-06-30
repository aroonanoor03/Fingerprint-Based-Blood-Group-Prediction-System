import os
import time
import joblib
import cv2
import numpy as np
from PIL import Image
import streamlit as st
import plotly.graph_objects as go
from src.preprocessing import preprocess_image
from src.feature_extraction import extract_hog_features

def is_valid_fingerprint(image_path):
    """
    Checks if the raw image resembles a fingerprint scan.
    We analyze block-level variance on the raw image (before equalization)
    along with global edge and structure density to detect non-fingerprints
    (blank files, noise, general photos, text, logos).
    """
    img_uint8 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_uint8 is None:
        return False, "Failed to load image."
        
    # Resize to standard size for uniform analysis
    img_uint8 = cv2.resize(img_uint8, (224, 224))
    
    # 1. Block-level variance analysis on RAW image (prevents equalization noise amplification)
    # Fingerprints contain active ridge textures across the entire image space.
    block_size = 16
    h, w = img_uint8.shape
    block_stds = []
    for r in range(0, h, block_size):
        for c in range(0, w, block_size):
            block = img_uint8[r:r+block_size, c:c+block_size]
            if block.shape == (block_size, block_size):
                block_stds.append(block.std())
    block_stds = np.array(block_stds)
    
    # Active blocks are those with standard deviation > 12.0 (on [0, 255] scale)
    active_fraction = np.mean(block_stds > 12.0)
    std_std = block_stds.std()
    
    # 2. Thresholding and transitions on median-blurred raw image
    img_blur = cv2.medianBlur(img_uint8, 3)
    thresh = cv2.adaptiveThreshold(
        img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    transitions = np.sum(np.diff(thresh) != 0)
    
    # 3. Sobel gradient mean
    sobelx = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    mag_mean = magnitude.mean()
    
    # 4. Laplacian variance
    lap_var = cv2.Laplacian(img_blur, cv2.CV_64F).var()
    
    # Validate block coverage (rejects centered emblem/logo on solid/empty backgrounds)
    if active_fraction < 0.60:
        return False, f"Texture coverage too low ({active_fraction*100:.1f}%). Fingerprint ridges must cover the scan area, whereas logos/graphics leave blank/solid boundaries."
        
    # Validate homogeneity of texture
    if std_std > 38.0:
        return False, f"Texture homogeneity too low (Block Spread: {std_std:.1f}). Fingerprints have a uniform ridge pattern, whereas text/emblems have highly variable contrast."
        
    # Validate edge density and contrast
    if not (100.0 <= mag_mean <= 250.0):
        return False, f"Invalid ridge contrast (Gradient Mean: {mag_mean:.1f}). Ensure the image contains clear ridge paths."
    if not (2500 <= transitions <= 9000):
        return False, f"Invalid ridge density (Transitions: {transitions}). Ensure the image is not noise, text, or a plain graphic."
    if not (500.0 <= lap_var <= 3500.0):
        return False, f"Invalid structural variance (Laplacian Var: {lap_var:.1f}). Distinct alternating ridge and valley patterns are required."
        
    return True, "Valid fingerprint scan."

# Set page config
st.set_page_config(
    page_title="Fingerprint Blood Group Predictor",
    page_icon="🩸",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS styling injection
st.markdown("""
<style>
    /* Main body background color */
    .stApp {
        background-color: #0e0e10;
        color: #e2e8f0;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    /* Title styling */
    h1 {
        background: linear-gradient(90deg, #ff4b4b, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        text-align: center;
    }
    
    /* Outer uploader container */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #1a1a1d !important;
        border: 2px dashed #ff4b4b !important;
        border-radius: 14px !important;
        padding: 20px !important;
        transition: 0.3s !important;
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        box-shadow: 0 0 15px rgba(255, 75, 75, 0.3) !important;
        border-color: #ff6b6b !important;
    }

    /* "Drag and drop" / label text */
    [data-testid="stFileUploaderDropzoneInstructions"] span,
    [data-testid="stFileUploaderDropzoneInstructions"] small {
        color: #ddd !important;
    }

    /* Upload button itself */
    [data-testid="stFileUploaderDropzone"] button {
        background: linear-gradient(90deg, #ff4b4b, #ff6b6b) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: 0.2s !important;
    }

    [data-testid="stFileUploaderDropzone"] button:hover {
        transform: scale(1.03) !important;
        box-shadow: 0 0 12px rgba(255,75,75,0.5) !important;
    }

    /* Label above uploader */
    [data-testid="stWidgetLabel"] p {
        color: #f5f5f5 !important;
        font-size: 16px !important;
        font-weight: 500 !important;
    }

    /* Uploaded file preview row */
    [data-testid="stFileUploaderFile"] {
        background-color: #1a1a1d !important;
        border: 1px solid #333 !important;
        border-radius: 8px !important;
        color: #ddd !important;
    }
    
    /* Styled buttons */
    .stButton>button {
        background: linear-gradient(90deg, #ff4b4b, #ff6b6b);
        color: white;
        border-radius: 10px;
        height: 3.2em;
        font-weight: 600;
        border: none;
        transition: 0.3s;
        box-shadow: 0 4px 10px rgba(255, 75, 75, 0.2);
    }
    
    .stButton>button:hover {
        transform: scale(1.01);
        box-shadow: 0 0 20px rgba(255, 75, 75, 0.6);
        color: white !important;
        border: none !important;
    }
    
    /* Target Streamlit image previews to apply borders and roundings */
    [data-testid="stImage"] img {
        border: 2px solid #ff4b4b;
        border-radius: 12px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    }
    
    /* Result card container with glowing keyframe animations */
    .result-card {
        background: radial-gradient(circle, rgba(255, 75, 75, 0.12), rgba(0, 0, 0, 0));
        border: 1px solid #ff4b4b;
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 25px;
        animation: glow 2.5s infinite alternate;
    }
    
    @keyframes glow {
        from { box-shadow: 0 0 6px rgba(255, 75, 75, 0.2); }
        to { box-shadow: 0 0 22px rgba(255, 75, 75, 0.55); }
    }
</style>
""", unsafe_allow_html=True)

# Sidebar layout
with st.sidebar:
    st.markdown("## 📊 Model Information")
    st.markdown("""
    This system uses a **Linear Support Vector Machine (SVM)** classifier trained on Histogram of Oriented Gradients (HOG) features.
    
    * **Training Dataset**: 6,000 scan images
    * **Feature Extractor**: HOG (6,084 dims)
    * **Test Partition**: 20% Stratified (1,200 scans)
    * **Model Test Accuracy**: **90.58%**
    * **Macro Precision**: **90.80%**
    * **Macro Recall**: **90.52%**
    """)
    
    st.markdown("---")
    st.markdown("## 🔍 How it Works")
    with st.expander("1. Image Preprocessing"):
        st.write("""
        Loads image, resizes to 224x224, converts to grayscale, applies a 3x3 median filter for sweat-pore noise removal, and uses histogram equalization to enhance contrast.
        """)
    with st.expander("2. Feature Extraction"):
        st.write("""
        Extracts local gradient orientation distributions (HOG) to mathematically represent fingerprint ridge flow patterns.
        """)
    with st.expander("3. SVM Classification"):
        st.write("""
        Computes distance vectors to decision hyperplanes to predict the class and estimate probability distributions.
        """)

# Header
st.markdown("<h1 style='text-align:center;'>🩸 Fingerprint Blood Group Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#aaa; margin-bottom: 25px;'>Upload a fingerprint scan to predict the associated blood group using SVM</p>", unsafe_allow_html=True)

model_path = "models/svm_model.pkl"

if not os.path.exists(model_path):
    st.error(f"⚠️ Model file not found at '{model_path}'. Please train the SVM model first.")
else:
    # Load model
    @st.cache_resource
    def load_best_model(path):
        return joblib.load(path)
        
    model = load_best_model(model_path)

    # Main uploader with icon in the label
    uploaded_file = st.file_uploader(
        "🩸 Upload a fingerprint scan image", 
        type=["bmp", "png", "jpg", "jpeg"],
        help="200MB max per file"
    )
    
    if uploaded_file is not None:
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        # 1. Preprocess and validate immediately upon upload
        preprocessed = preprocess_image(temp_file_path)
        is_valid = False
        validation_msg = ""
        
        if preprocessed is not None:
            is_valid, validation_msg = is_valid_fingerprint(temp_file_path)
            
        # Display immediate feedback caption below the uploader box
        if is_valid:
            st.markdown("<div style='color: #4ade80; font-size: 0.9rem; font-weight: 500; margin-top: -10px; margin-bottom: 15px;'>✅ Fingerprint loaded</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='color: #ef4444; font-size: 0.9rem; font-weight: 500; margin-top: -10px; margin-bottom: 15px;'>❌ Invalid fingerprint scan</div>", unsafe_allow_html=True)
            
        # Layout columns to display image preview
        col1, col2 = st.columns([1, 2.2])
        with col1:
            st.write("")  # padding
            st.image(
                Image.open(uploaded_file), 
                caption="Uploaded Scan File", 
                use_container_width=True
            )
            
        with col2:
            st.write("### Analysis Controls")
            
            if not is_valid:
                st.warning("Prediction disabled: The uploaded file is not a valid fingerprint scan.")
                # Show the warning message detailing the failure
                st.markdown(f"""
                <div style="background-color: rgba(239, 68, 68, 0.1); border: 1px solid #ef4444; border-radius: 12px; padding: 15px; color: #f87171; margin-top: 10px; box-shadow: 0 4px 10px rgba(239, 68, 68, 0.15);">
                    <h5 style="color: #ef4444; margin-top: 0; margin-bottom: 5px;">⚠️ Validation Error</h5>
                    <p style="margin: 0; font-size: 0.9rem; line-height: 1.4;">{validation_msg}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Render disabled button
                st.button("Predict Blood Group", type="primary", use_container_width=True, disabled=True)
            else:
                st.info("Fingerprint loaded successfully. Click below to predict the blood group.")
                
                # Predict trigger
                if st.button("Predict Blood Group", type="primary", use_container_width=True):
                    with st.spinner("Analyzing ridge patterns... 🔬"):
                        # Simulate visual wait for processing feedback
                        time.sleep(1.2)
                        
                        # 2. Extract HOG
                        features = extract_hog_features(preprocessed)
                        features_2d = features.reshape(1, -1)
                        
                        # 3. Predict class
                        prediction = model.predict(features_2d)
                        predicted_group = prediction[0]
                        
                        # 4. Compute probabilities (using softmax over decision function values)
                        decision_values = model.decision_function(features_2d)[0]
                        # Softmax scaling for pseudo-probabilities
                        exp_vals = np.exp(decision_values - np.max(decision_values))
                        probabilities = exp_vals / np.sum(exp_vals)
                        
                        # Map to class dictionary
                        classes_list = list(model.classes_)
                        prob_dict = {classes_list[i]: float(probabilities[i]) for i in range(len(classes_list))}
                        
                        # Display Custom HTML Result Card
                        st.markdown(f"""
                        <div class="result-card">
                            <p style="color:#ff6b6b; letter-spacing:2.5px; font-size:12px; font-weight: bold; margin:0 0 5px 0;">PREDICTED BLOOD GROUP</p>
                            <h1 style="color:white; font-size:52px; margin:0; line-height:1.1; text-shadow: 0 0 10px rgba(255, 75, 75, 0.4);">{predicted_group}</h1>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display Confidence Bar Chart
                        st.write("#### 📈 Prediction Confidence Profile")
                        fig = go.Figure(go.Bar(
                            x=list(prob_dict.keys()),
                            y=list(prob_dict.values()),
                            marker_color='#ff4b4b',
                            text=[f"{v*100:.1f}%" for v in prob_dict.values()],
                            textposition='outside'
                        ))
                        fig.update_layout(
                            template="plotly_dark",
                            height=250,
                            margin=dict(t=20, b=10, l=10, r=10),
                            xaxis_title="Blood Group Class",
                            yaxis_title="Confidence Probability",
                            yaxis_range=[0, 1.15],
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
        # Cleanup temporary files
        try:
            os.remove(temp_file_path)
        except Exception:
            pass


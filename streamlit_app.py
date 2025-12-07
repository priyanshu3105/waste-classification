

import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import json
import os
import time
import base64
import io

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Smart Waste Management",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #2ecc71;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .organic-card {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 3px solid #28a745;
    }
    .recyclable-card {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border: 3px solid #17a2b8;
    }
    .non-organic-card {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border: 3px solid #dc3545;
    }
    .confidence-text {
        font-size: 1.3rem;
        font-weight: bold;
        margin-top: 0.5rem;
    }
    .image-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        grid-gap: 25px;
        margin-top: 30px;
        padding: 20px;
    }
    .grid-item {
        text-align: center;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        transition: transform 0.2s;
    }
    .grid-item:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.2);
    }
    .grid-item img {
        border-radius: 10px;
        margin-bottom: 15px;
        width: 100%;
        max-width: 250px;
        height: 250px;
        object-fit: cover;
        border: 2px solid #ddd;
    }
    .preview-container {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        padding: 15px 0;
        justify-content: flex-start;
    }
    .preview-thumbnail {
        width: 120px;
        height: 120px;
        border-radius: 12px;
        object-fit: cover;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        transition: all 0.2s ease;
        cursor: pointer;
    }
    .preview-thumbnail:hover {
        transform: translateY(-4px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.25);
    }
    .category-badge {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 20px;
        font-size: 1.1rem;
        font-weight: bold;
        margin: 10px 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .organic-badge {
        background-color: #28a745;
        color: white;
    }
    .recyclable-badge {
        background-color: #17a2b8;
        color: white;
    }
    .non-organic-badge {
        background-color: #dc3545;
        color: white;
    }
    .confidence-bar {
        width: 100%;
        background-color: #e9ecef;
        border-radius: 10px;
        height: 25px;
        margin: 10px 0;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .description-text {
        color: #495057;
        font-size: 0.9rem;
        margin-top: 10px;
        padding: 10px;
        background-color: rgba(255,255,255,0.7);
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# CACHED MODEL LOADING
# ---------------------------------------------------------
@st.cache_resource
def load_model(model_path):
    try:
        # Check model file modification time to force reload if changed
        model_mtime = os.path.getmtime(model_path)
        model = keras.models.load_model(model_path)
        st.sidebar.caption(f"Model loaded: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(model_mtime))}")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_stats():
    try:
        with open("results/evaluation_report.json", "r") as f:
            return json.load(f)
    except:
        return None

# Class info
CLASS_INFO = {
    'organic': {
        'emoji': '🥬',
        'color': 'organic',
        'description': 'Biodegradable materials such as food waste.',
        'examples': 'Fruit peels, vegetables, leaves, leftovers.',
    },
    'recyclable': {
        'emoji': '♻️',
        'color': 'recyclable',
        'description': 'Materials that can be recycled and reused.',
        'examples': 'Plastic bottles, cans, cardboard, glass.',
    },
    'non_organic': {
        'emoji': '🗑️',
        'color': 'non-organic',
        'description': 'General waste that cannot be composted.',
        'examples': 'Clothes, shoes, dirty plastic, mixed waste.',
    }
}

# ---------------------------------------------------------
# UTILITY FUNCTIONS (defined before use)
# ---------------------------------------------------------
def image_to_base64(img):
    """Convert PIL image to base64 string for embedding in HTML"""
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

def preprocess_image(img, target_size=(224, 224)):
    """Preprocess image to match model input requirements - FIXED to 160x160"""
    img = img.resize(target_size)
    img_arr = np.array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)
    return img_arr

def predict_class(model, img_array):
    """Make prediction and return class, confidence, and probabilities"""
    preds = model.predict(img_array, verbose=0)
    class_names = ['non_organic', 'organic', 'recyclable']
    idx = np.argmax(preds[0])
    pred_class = class_names[idx]
    conf = preds[0][idx] * 100
    return pred_class, conf, preds[0]


# ---------------------------------------------------------
# UI HEADER
# ---------------------------------------------------------
st.markdown('<div class="main-header">♻️ Smart Waste Management</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered waste classifier with real-time predictions</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.header("📊 Model Settings")

    model_choice = st.radio("Choose model:", ["Advanced (ResNet50)"])

    model_path = (
        "models/resnet50_advanced_model.h5"
    )

    stats = load_stats()

    if stats:
        key = "advanced_model" if "Advanced" in model_choice else "baseline_model"
        st.metric("Accuracy", f"{stats[key]['accuracy']*100:.2f}%")

    st.markdown("---")
    
    if st.button("🔄 Reload Model", help="Clear cache and reload the model"):
        st.cache_resource.clear()
        st.rerun()
    
    st.write("Upload single images or multiple images using drag & drop.")


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
model = load_model(model_path)
if model:
    st.success(f"Model Loaded: {model_choice}")
else:
    st.stop()

# ---------------------------------------------------------
# IMAGE UPLOADER 
# ---------------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload Image(s)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# ---------------------------------------------------------
# IF IMAGES UPLOADED → SHOW PREVIEW + PREDICT BUTTON
# ---------------------------------------------------------
if uploaded_files:

    st.subheader(f"📤 {len(uploaded_files)} Image(s) Uploaded")
    
    num_cols = min(len(uploaded_files), 6)
    preview_cols = st.columns(num_cols)
    
    for idx, file in enumerate(uploaded_files):
        if idx < num_cols:  # Show max 6 previews
            with preview_cols[idx]:
                img = Image.open(file)
                st.image(img, use_container_width=True, caption=file.name[:15])
    
    if len(uploaded_files) > 6:
        st.caption(f"... and {len(uploaded_files) - 6} more image(s)")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button(
            "🔍 Classify Waste Images",
            type="primary",
            use_container_width=True
        )
    
    if predict_button:
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🖼 Classification Results")
        
        progress = st.progress(0)
        status_text = st.empty()
        step = 100 / len(uploaded_files)

        # Process images
        results = []
        for i, file in enumerate(uploaded_files):
            status_text.text(f"Processing image {i+1}/{len(uploaded_files)}...")
            img = Image.open(file)
            img_arr = preprocess_image(img)

            # progress bar
            progress.progress(int((i + 1) * step))

            pred_class, conf, probs = predict_class(model, img_arr)
            results.append((img, pred_class, conf))
        
        progress.empty()
        status_text.empty()

        cols_per_row = 3
        for i in range(0, len(results), cols_per_row):
            cols = st.columns(cols_per_row)
            
            for j in range(cols_per_row):
                if i + j < len(results):
                    img, pred_class, conf = results[i + j]
                    info = CLASS_INFO[pred_class]
                    
                    card_class = f"{pred_class.replace('_', '-')}-card"
                    badge_class = f"{pred_class.replace('_', '-')}-badge"
                    
                    if pred_class == 'organic':
                        bar_color = '#28a745'
                    elif pred_class == 'recyclable':
                        bar_color = '#17a2b8'
                    else:
                        bar_color = '#dc3545'
                    
                    category_name = pred_class.replace('_', ' ').title()
                    if pred_class == 'non_organic':
                        category_name = 'Non-Organic'
                    
                    img_b64 = image_to_base64(img)
                    
                    with cols[j]:
                        # HTML for single card
                        card_html = f'''
                        <div class="grid-item {card_class}">
                            <img src="data:image/png;base64,{img_b64}"/>
                            <div style="font-size: 3rem; margin: 10px 0;">{info['emoji']}</div>
                            <div class="category-badge {badge_class}">{category_name}</div>
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: {conf}%; background-color: {bar_color};">
                                    {conf:.1f}% Confidence
                                </div>
                            </div>
                            <div class="description-text">{info['description']}</div>
                            <div style="margin-top: 10px; font-size: 0.85rem; color: #6c757d;">
                                <b>Examples:</b> {info['examples']}
                            </div>
                        </div>
                        '''
                        st.markdown(card_html, unsafe_allow_html=True)
        
        st.success(f"✅ Successfully classified {len(results)} image(s)!")

else:
    st.info("👆 Drag and drop images above or click to upload.")

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; color:#7f8c8d;">
    Smart Waste Management System • Built with Streamlit & TensorFlow
    </div>
    """,
    unsafe_allow_html=True,
)
# 🗑️ Smart Waste Classification System
### Transfer Learning with ResNet50

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/tensorflow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Overview

A deep learning-based waste classification system that leverages **ResNet50 transfer learning** to accurately categorize waste into three primary categories: **Organic**, **Recyclable**, and **Non-Organic**. This system achieves **77-83% accuracy** through a sophisticated two-phase training approach combining frozen feature extraction with fine-tuning.

### Key Features
✨ Two-phase transfer learning strategy  
✨ ResNet50 architecture pre-trained on ImageNet  
✨ Comprehensive data augmentation pipeline  
✨ Interactive Streamlit web application  
✨ Detailed performance metrics and visualizations  
✨ Easy deployment and reproducibility  

---

## 🎯 Model Performance

| Metric | Value |
|--------|-------|
| **Architecture** | ResNet50 + Custom Dense Layers |
| **Expected Accuracy** | 77-83% |
| **Training Strategy** | Frozen Base → Fine-tuning |
| **Input Size** | 224 × 224 × 3 |
| **Classes** | 3 (Organic, Recyclable, Non-Organic) |
| **Training Time** | ~2-3 hours on GPU |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- GPU recommended (CPU will work but slower)
- ~5GB disk space for dataset and models

### Installation
```bash
# 1. Clone the repository
git clone https://github.com/priyanshu3105/waste-classification.git
cd waste-classification

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 📥 Download Required Files

### 📦 Dataset
**⚠️ Required for training/evaluation**

- **Download Link:** [Google Drive - Dataset](https://drive.google.com/drive/folders/1ikRc06pV3Qer3_GD2mbXSDJv-gubw5hB?usp=drive_link)
- **Size:** ~240 MB (compressed)
- **Extract to:** `data/garbage_classification/`
- **Structure:**
```
  data/garbage_classification/
  ├── battery/
  ├── biological/
  ├── brown-glass/
  ├── cardboard/
  ├── clothes/
  ├── green-glass/
  ├── metal/
  ├── paper/
  ├── plastic/
  ├── shoes/
  ├── trash/
  └── white-glass/
```

### 🤖 Pre-trained Models
**⚠️ Required for running the app without training**

- **Download Link:** [Google Drive - Models](https://drive.google.com/drive/folders/1yN9aFpoYq_92iuYUwDXhpznuP9kSwxYr?usp=drive_link)
- **Files:**
  - `baseline_fast_model.h5` (~1.4 MB)
  - `resnet50_advanced_model.h5` (~250 MB)
- **Place in:** `models/` directory

---

## 📂 Project Structure
```
waste-classification/
│
├── data/
│   ├── garbage_classification/     # Raw dataset (download separately)
│   └── processed/                  # Processed data (generated)
│
├── models/                         # Pre-trained models (download separately)
│   ├── baseline_fast_model.h5
│   └── resnet50_advanced_model.h5
│
├── notebooks/
│   └── Evaluation_graphs.ipynb    # Jupyter notebook for analysis
│
├── results/
│   ├── plots/                     # Training visualizations
│   ├── advanced_history_resnet50.json
│   └── evaluation_report.json
│
├── 1_data_processing.py           # Data preprocessing pipeline
├── 2_baseline_model.py            # Baseline model training
├── 4_evaluation.py                # Model evaluation
├── 5_advanced_model2.0.py         # ResNet50 training
├── streamlit_app.py               # Web application
├── verify.py                      # Verification script
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🔧 Usage

### Option 1: Run Pre-trained Model (Recommended)
```bash
# Download models first, then:
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

### Option 2: Train from Scratch
```bash
# Step 1: Process the dataset
python 1_data_processing.py

# Step 2: Train baseline model (optional)
python 2_baseline_model.py

# Step 3: Train advanced ResNet50 model
python 5_advanced_model2.0.py

# Step 4: Evaluate models
python 4_evaluation.py

# Step 5: Run the app
streamlit run streamlit_app.py
```

---

## 🏗️ Model Architecture
```
Input (224×224×3)
    ↓
ResNet50 Base (ImageNet Weights)
    ↓ [Frozen - Phase 1]
Convolutional Blocks (Layers 1-100)
    ↓
Fine-tuned Layers (Layers 101-175)
    ↓ [Trainable - Phase 2]
Global Average Pooling (2048 features)
    ↓
Batch Normalization
    ↓
Dense (512 units, ReLU)
    ↓
Dropout (0.4)
    ↓
Dense (3 units, Softmax)
    ↓
Output: [Organic, Recyclable, Non-Organic]
```

---

## 🎓 Training Details

### Two-Phase Training Strategy

**Phase 1: Feature Extraction (10 epochs)**
- Base ResNet50 layers: **Frozen**
- Only custom top layers trained
- Learning Rate: 0.0005 (CosineDecay)
- Optimizer: AdamW with weight decay

**Phase 2: Fine-tuning (20 epochs)**
- Layers 101-175: **Trainable**
- Earlier layers remain frozen
- Learning Rate: 1e-5
- Fine-grained feature adaptation

### Data Augmentation
```python
Augmentation Techniques:
├── Rotation: ±15°
├── Width Shift: 10%
├── Height Shift: 10%
├── Zoom Range: 10%
├── Horizontal Flip: Yes
└── Brightness: 0.9 - 1.1
```

### Hyperparameters
```python
CONFIG = {
    'img_size': (224, 224),
    'batch_size': 32,
    'initial_epochs': 10,
    'fine_tune_epochs': 20,
    'validation_split': 0.2,
    'num_classes': 3,
    'fine_tune_at': 100,
}
```

---

## 📊 Results & Visualizations

The training process generates several visualization plots in `results/plots/`:

- **Training History:** Loss and accuracy curves
- **Confusion Matrix:** Classification performance
- **ROC Curves:** Class-wise performance
- **Precision-Recall Curves:** Detailed metrics
- **Misclassification Examples:** Error analysis

---

## 🔬 Reproducibility

### Random Seeds
```python
np.random.seed(42)
tf.random.set_seed(42)
```

### Environment
- TensorFlow 2.13.0
- Python 3.8+
- CUDA-enabled GPU (recommended)

### Expected Results
- Validation Accuracy: 77-83%
- Training Time: 2-3 hours (GPU) / 8-12 hours (CPU)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

### Areas for Improvement
- [ ] Add more waste categories
- [ ] Implement real-time video classification
- [ ] Optimize for mobile deployment
- [ ] Add explainability (Grad-CAM)
- [ ] Multi-language support

---

## 📝 Citation

If you use this project in your research or work, please cite:
```bibtex
@misc{waste-classification-2024,
  author = {Priyanshu Chatterjee},
  title = {Smart Waste Classification using ResNet50 Transfer Learning},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/priyanshu3105/waste-classification}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Priyanshu Chatterjee**

- GitHub: [@priyanshu3105](https://github.com/priyanshu3105)
- Repository: [waste-classification](https://github.com/priyanshu3105/waste-classification)

---

## 🙏 Acknowledgments

- ResNet50 architecture from [He et al., 2016](https://arxiv.org/abs/1512.03385)
- Transfer learning techniques from TensorFlow/Keras
- ImageNet pre-trained weights
- Open-source deep learning community

---

## 📞 Support

For questions, issues, or suggestions:
- Open an issue on GitHub
- Contact via GitHub profile

---

## 🔄 Version History

- **v1.0.0** (2024) - Initial release
  - ResNet50 transfer learning implementation
  - Two-phase training strategy
  - Streamlit web application
  - Comprehensive evaluation metrics

---

**⭐ If you find this project useful, please consider giving it a star!**

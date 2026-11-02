# 🔬 SilicaSense: Silica Gel Hydration Phase & Color Classifier

An end-to-end deep learning and computer vision pipeline for automated condition classification of **Blue** and **Orange Silica Gel** desiccants based on chemical indicator color changes.

Given an input photo of silica gel beads, the model automatically detects:
1. **Gel Type**: Blue Silica Gel or Orange Silica Gel
2. **Current Phase**: **Unsaturated** *(Fresh / Dry)*, **Semi-Saturated** *(Partially Absorbed)*, or **Saturated** *(Exhausted)*
3. **Current Colour**: Measured dominant color name, Hex code (`#...`), and RGB values.

---

## 🧪 Chemical Indicator Color Transition Principles

Silica gel beads are infused with moisture indicators that shift chromaticity as they absorb water vapor from the atmosphere:

| Gel Type | Chemical Indicator | Phase | Hydration State | Visual Indicator Color |
| :--- | :--- | :---: | :--- | :--- |
| **Blue Silica Gel** | Cobalt Chloride ($CoCl_2$) | `Unsaturated` | Fresh / Active (Dry) | **Deep Royal / Cobalt Blue** |
| **Blue Silica Gel** | Cobalt Chloride ($CoCl_2$) | `Semi-Saturated` | Absorbing moisture | **Purple / Violet / Mauve** |
| **Blue Silica Gel** | Cobalt Chloride ($CoCl_2$) | `Saturated` | Fully exhausted | **Pink / Light Rose** |
| **Orange Silica Gel** | Organic / Iron indicator | `Unsaturated` | Fresh / Active (Dry) | **Bright Orange / Amber** |
| **Orange Silica Gel** | Organic / Iron indicator | `Semi-Saturated` | Absorbing moisture | **Olive / Yellow-Green** |
| **Orange Silica Gel** | Organic / Iron indicator | `Saturated` | Fully exhausted | **Dark Green / Teal / Blackish** |

> **⚠️ Critical Augmentation Note**: Because the classification signal is strictly color-based, standard aggressive color jitter (hue shifts, channel inversion, solarize) is strictly prohibited. The augmentation pipeline locks **`Hue = 0.0`** while leveraging physical $360^\circ$ rotational invariance, flips, perspective jitter, and sensor noise.

---

## 📂 Project Architecture & Repository Structure

```text
project_transformer/
├── blue_silicagel/                    # Blue silica gel dataset
│   ├── initial/                       # Unsaturated / Dry samples (50 images)
│   ├── unsaturated/                   # Semi-saturated / Absorbing samples (50 images)
│   └── saturated/                     # Saturated / Exhausted samples (50 images)
├── orange_silicagel/                  # Orange silica gel dataset
│   ├── initial/                       # Unsaturated / Dry samples (50 images)
│   ├── unsaturated/                   # Semi-saturated / Absorbing samples (50 images)
│   └── saturated/                     # Saturated / Exhausted samples (50 images)
│
├── silica_gel_data_augmentation.ipynb # Dataset balancing & augmentation notebook
├── silica_gel_model_training.ipynb    # ResNet-50 training, validation & evaluation notebook
├── best_silicagel_model.pth           # Saved trained model weights (~98.5 MB)
│
├── predict.py                         # Standalone Python CLI & inference module
├── app.py                             # Flask web server backend
│
├── templates/
│   └── index.html                     # Minimal, focused web inference frontend
├── static/
│   ├── css/style.css                  # Clean, modern styling
│   └── js/app.js                      # Image upload & prediction client logic
│
└── requirements.txt                   # Project dependencies
```

---

## 📊 Dataset Balancing & Model Training

### 1. Data Augmentation
- **Initial Dataset**: 37 original raw photographs across the 6 condition folders.
- **Augmentation Pipeline**: Implemented with `torchvision.transforms.v2` in [`silica_gel_data_augmentation.ipynb`](silica_gel_data_augmentation.ipynb).
- **Target**: Balanced to **exactly 50 images per condition folder** (300 total balanced images). Augmented images are saved directly inside their respective subfolders with the `aug_` prefix.

### 2. Model Training & Validation Split
- **Split Strategy**: Stratified **80% Training (240 samples)** and **20% Validation (60 samples)** — exactly 40 train and 10 val images per class.
- **Architecture**: Pretrained **ResNet-50** with a custom 6-class classifier head:
  $$\text{ResNet-50 Feature Extractor (2048)} \to \text{Dropout}(0.3) \to \text{Linear}(2048, 512) \to \text{ReLU} \to \text{Dropout}(0.3) \to \text{Linear}(512, 6)$$
- **Hardware Used**: NVIDIA GeForce RTX 4050 Laptop GPU (CUDA).
- **Optimizer & Scheduler**: AdamW ($lr=10^{-4}$, weight decay $=10^{-2}$) + Cosine Annealing scheduler across 15 epochs.
- **Performance**: **100% Validation Accuracy** (60/60 correct predictions on unseen validation images).
- **Checkpoint**: Saved to [`best_silicagel_model.pth`](best_silicagel_model.pth).

---

## 🚀 Quick Start Guide

### 1. Installation
Clone or navigate to the repository directory and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the Web Inference Frontend
Start the local web server:
```bash
python app.py
```
Open your browser and navigate to:
👉 **`http://127.0.0.1:5000`**

- **Upload Box**: Drag and drop any silica gel photo, click to browse, or press <kbd>Ctrl+V</kbd> to paste from clipboard.
- **Instant Output**:
  - **Gel Type**: Blue Silica Gel or Orange Silica Gel
  - **Current Phase**: Unsaturated, Semi-Saturated, or Saturated
  - **Current Colour**: Characteristic color name, real color swatch, and exact Hex code.

---

## 💻 Command Line Inference (`predict.py`)

You can also run inference on any image directly from the terminal:

```bash
python predict.py --image "path/to/any_silica_gel_photo.jpeg"
```

### Example Terminal Output:
```text
Loaded model on device: cuda
=================================================================
SILICA GEL CLASSIFICATION RESULT
=================================================================
Image File    : path/to/sample.jpeg
Gel Type      : Blue Silica Gel
Condition     : Unsaturated (Fresh/Dry)
Confidence    : 94.08%
Expected Hue  : Deep Royal/Cobalt Blue
Measured Color: RGB=(64, 73, 100), Hex=#404964, Hue=225.0°
-----------------------------------------------------------------
Class Probabilities:
  • blue_unsaturated        :  94.1%
  • blue_semi_saturated     :   1.1%
  • blue_saturated          :   1.5%
  • orange_unsaturated      :   1.0%
  • orange_semi_saturated   :   1.4%
  • orange_saturated        :   0.9%
=================================================================
```

### Optional Flags:
- `--save-plot output_plot.png`: Saves a visualization figure showing the input image, extracted color swatch, and probability bar chart.
- `--no-plot`: Suppresses the interactive matplotlib GUI window.

---

## 🐍 Python API Integration

To use the model in your own custom Python scripts or pipelines:

```python
from predict import predict_silica_gel

# Analyze any image file or PIL Image object
result = predict_silica_gel("path/to/my_silica_gel.jpg")

print("Gel Type      :", result["gel_type"])  # 'Blue' or 'Orange'
print("Phase         :", result["condition"])  # 'Unsaturated', 'Semi-Saturated', 'Saturated'
print("Confidence    :", result["confidence_pct"], "%")
print("Dominant Color:", result["dominant_color"]["hex"])  # e.g., '#404964'
```

---

## 📓 Jupyter Notebooks

1. **[`silica_gel_data_augmentation.ipynb`](silica_gel_data_augmentation.ipynb)**:
   - Contains dataset inventory scan, distribution audit, augmentation pipeline definition, visual validation grids, and in-place expansion code.
2. **[`silica_gel_model_training.ipynb`](silica_gel_model_training.ipynb)**:
   - Contains the 80/20 stratified split, ResNet-50 training loop, loss/accuracy curves, validation confusion matrix, and interactive file upload widget.

---

## 📄 License
This project is open-source and intended for humidity indicator monitoring, desiccant condition tracking, and computer vision classification research.

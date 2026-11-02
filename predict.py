"""
Silica Gel Condition Classifier - Standalone Inference Script
Predicts Gel Type (Blue or Orange) and Saturation Condition (Unsaturated, Semi-Saturated, Saturated)
from an input image file based on chemical indicator color changes.
"""

import os
import sys
import argparse
import colorsys
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import matplotlib.pyplot as plt

CLASS_DEFS = [
    {'id': 0, 'name': 'blue_unsaturated', 'gel': 'Blue', 'condition': 'Unsaturated (Fresh/Dry)', 'color': 'Deep Royal/Cobalt Blue'},
    {'id': 1, 'name': 'blue_semi_saturated', 'gel': 'Blue', 'condition': 'Semi-Saturated (Partially Absorbed)', 'color': 'Purple/Violet/Mauve'},
    {'id': 2, 'name': 'blue_saturated', 'gel': 'Blue', 'condition': 'Saturated (Exhausted)', 'color': 'Pink/Light Rose'},
    {'id': 3, 'name': 'orange_unsaturated', 'gel': 'Orange', 'condition': 'Unsaturated (Fresh/Dry)', 'color': 'Bright Orange/Amber'},
    {'id': 4, 'name': 'orange_semi_saturated', 'gel': 'Orange', 'condition': 'Semi-Saturated (Partially Absorbed)', 'color': 'Olive/Yellow-Green'},
    {'id': 5, 'name': 'orange_saturated', 'gel': 'Orange', 'condition': 'Saturated (Exhausted)', 'color': 'Dark Green/Teal/Blackish'}
]

CLASS_NAMES = [c['name'] for c in CLASS_DEFS]
CLASS_DISPLAY = [f"{c['gel']} - {c['condition']}" for c in CLASS_DEFS]

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

infer_transforms = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

def load_model(weights_path=None, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    if weights_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(script_dir, 'best_silicagel_model.pth')
        
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {weights_path}. Please train the model first.")

    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, 6)
    )
    
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device

def analyze_image_color(pil_img):
    img_rgb = pil_img.convert('RGB')
    arr = np.array(img_rgb)
    mean_rgb = arr.mean(axis=(0, 1)).astype(int)
    r_norm, g_norm, b_norm = mean_rgb[0]/255.0, mean_rgb[1]/255.0, mean_rgb[2]/255.0
    h, s, v = colorsys.rgb_to_hsv(r_norm, g_norm, b_norm)
    return {
        'mean_rgb': tuple(int(x) for x in mean_rgb),
        'hex': '#{:02x}{:02x}{:02x}'.format(int(mean_rgb[0]), int(mean_rgb[1]), int(mean_rgb[2])),
        'h_deg': round(float(h * 360), 1),
        's_pct': round(float(s * 100), 1),
        'v_pct': round(float(v * 100), 1)
    }

def predict_silica_gel(image_path_or_pil, model=None, device=None, show_plot=False, save_plot_path=None):
    if model is None:
        model, device = load_model(device=device)
    if device is None:
        device = next(model.parameters()).device

    if isinstance(image_path_or_pil, str):
        if not os.path.exists(image_path_or_pil):
            raise FileNotFoundError(f"File not found: {image_path_or_pil}")
        pil_img = Image.open(image_path_or_pil).convert('RGB')
    elif isinstance(image_path_or_pil, Image.Image):
        pil_img = image_path_or_pil.convert('RGB')
    else:
        raise ValueError("Input must be a file path string or PIL.Image instance")

    tensor = infer_transforms(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx] * 100.0)

    pred_info = CLASS_DEFS[pred_idx]
    color_info = analyze_image_color(pil_img)

    result = {
        'gel_type': pred_info['gel'],
        'condition': pred_info['condition'],
        'class_name': pred_info['name'],
        'confidence_pct': round(confidence, 2),
        'characteristic_color': pred_info['color'],
        'dominant_color': color_info,
        'all_probabilities': {c['name']: round(float(p) * 100.0, 2) for c, p in zip(CLASS_DEFS, probs)}
    }

    if show_plot or save_plot_path:
        fig = plt.figure(figsize=(13, 4))
        
        # 1. Image
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(pil_img)
        ax1.set_title("Input Image", fontsize=11, fontweight='bold')
        ax1.axis('off')

        # 2. Color Swatch
        ax2 = fig.add_subplot(1, 3, 2)
        swatch = np.full((100, 100, 3), color_info['mean_rgb'], dtype=np.uint8)
        ax2.imshow(swatch)
        ax2.set_title("Dominant Color", fontsize=11, fontweight='bold')
        ax2.axis('off')
        text_desc = (
            f"Hex: {color_info['hex']}\\n"
            f"RGB: {color_info['mean_rgb']}\\n"
            f"Hue: {color_info['h_deg']}°\\n"
            f"Saturation: {color_info['s_pct']}%\\n"
            f"Brightness: {color_info['v_pct']}%"
        )
        ax2.text(50, 125, text_desc, ha='center', fontsize=10, family='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", edgecolor="#ccc"))
        ax2.set_ylim(155, -10)

        # 3. Bar chart
        ax3 = fig.add_subplot(1, 3, 3)
        bar_colors = ['#1f77b4' if 'blue' in name else '#ff7f0e' for name in CLASS_NAMES]
        y_pos = np.arange(len(CLASS_NAMES))
        bars = ax3.barh(y_pos, probs * 100.0, color=bar_colors, edgecolor='black', alpha=0.85)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(CLASS_DISPLAY, fontsize=8)
        ax3.invert_yaxis()
        ax3.set_xlabel('Probability (%)')
        ax3.set_xlim(0, 100)
        ax3.grid(axis='x', linestyle=':', alpha=0.5)
        bars[pred_idx].set_edgecolor('red')
        bars[pred_idx].set_linewidth(2)
        ax3.set_title(f"Prediction: {pred_info['gel']} - {pred_info['condition']}\\nConfidence: {confidence:.1f}%",
                      fontsize=10, fontweight='bold')

        plt.tight_layout()
        if save_plot_path:
            plt.savefig(save_plot_path, dpi=200, bbox_inches='tight')
            print(f"Plot saved to: {save_plot_path}")
        if show_plot:
            plt.show()
        plt.close()

    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Classify Silica Gel Condition (Unsaturated, Semi-Saturated, Saturated)")
    parser.add_argument('--image', type=str, required=True, help="Path to input image file")
    parser.add_argument('--weights', type=str, default=None, help="Path to model weights file (.pth)")
    parser.add_argument('--save-plot', type=str, default=None, help="Path to save result visualization figure (.png)")
    parser.add_argument('--no-plot', action='store_true', help="Do not display interactive plot")

    args = parser.parse_args()

    model, device = load_model(weights_path=args.weights)
    print(f"Loaded model on device: {device}")

    res = predict_silica_gel(
        args.image,
        model=model,
        device=device,
        show_plot=not args.no_plot,
        save_plot_path=args.save_plot
    )

    print("=" * 65)
    print("SILICA GEL CLASSIFICATION RESULT")
    print("=" * 65)
    print(f"Image File    : {args.image}")
    print(f"Gel Type      : {res['gel_type']} Silica Gel")
    print(f"Condition     : {res['condition']}")
    print(f"Confidence    : {res['confidence_pct']}%")
    print(f"Expected Hue  : {res['characteristic_color']}")
    print(f"Measured Color: RGB={res['dominant_color']['mean_rgb']}, Hex={res['dominant_color']['hex']}, Hue={res['dominant_color']['h_deg']}°")
    print("-" * 65)
    print("Class Probabilities:")
    for c_name, prob in res['all_probabilities'].items():
        print(f"  • {c_name:<24}: {prob:5.1f}%")
    print("=" * 65)

"""
Silica Gel Condition Classifier - Web Inference Server
Flask backend serving the interactive inference UI and API.
"""

import os
import io
import base64
from flask import Flask, render_template, request, jsonify, send_file, abort
from PIL import Image
import torch

from predict import load_model, predict_silica_gel, CLASS_DEFS

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20MB max upload

# Global model cache
print("Initializing model on GPU/CPU...")
MODEL, DEVICE = load_model()
print(f"Model ready on {DEVICE}!")

SAMPLE_IMAGES = [
    {
        'id': 'blue_init',
        'label': 'Blue - Fresh / Dry',
        'rel_path': 'blue_silicagel/initial/WhatsApp Image 2026-10-23 at 4.14.45 PM (2).jpeg',
        'tag': 'Fresh'
    },
    {
        'id': 'blue_unsat',
        'label': 'Blue - Semi-Saturated',
        'rel_path': 'blue_silicagel/unsaturated/WhatsApp Image 2026-10-23 at 4.15.04 PM (2).jpeg',
        'tag': 'Absorbing'
    },
    {
        'id': 'blue_sat',
        'label': 'Blue - Saturated',
        'rel_path': 'blue_silicagel/saturated/WhatsApp Image 2026-10-23 at 4.14.48 PM (2).jpeg',
        'tag': 'Exhausted'
    },
    {
        'id': 'orange_init',
        'label': 'Orange - Fresh / Dry',
        'rel_path': 'orange_silicagel/initial/WhatsApp Image 2026-10-23 at 4.14.47 PM (1).jpeg',
        'tag': 'Fresh'
    },
    {
        'id': 'orange_unsat',
        'label': 'Orange - Semi-Saturated',
        'rel_path': 'orange_silicagel/unsaturated/WhatsApp Image 2026-10-23 at 4.14.56 PM (1).jpeg',
        'tag': 'Absorbing'
    },
    {
        'id': 'orange_sat',
        'label': 'Orange - Saturated',
        'rel_path': 'orange_silicagel/saturated/WhatsApp Image 2026-10-23 at 4.14.44 PM (1).jpeg',
        'tag': 'Exhausted'
    }
]

def get_recommendation(gel_type, condition):
    condition_lower = condition.lower()
    if 'unsaturated' in condition_lower and 'semi' not in condition_lower:
        return {
            'status_level': 'success',
            'status_title': 'Active Desiccant (Ready to Use)',
            'description': f'This {gel_type} silica gel is completely dry and active. Moisture absorbing capacity is at maximum (~100%).',
            'action': 'No action required. Safe to store in electronic enclosures, filament boxes, or packaging.'
        }
    elif 'semi' in condition_lower:
        return {
            'status_level': 'warning',
            'status_title': 'Partially Absorbed (Active Drying in Progress)',
            'description': f'The chemical indicator has shifted, showing partial moisture uptake (~30-60% capacity utilized).',
            'action': 'Still functional, but monitor closely. Prepare for regeneration if used in critical low-humidity environments.'
        }
    else:  # Saturated
        temp = '120°C (250°F) for 2 to 3 hours' if gel_type == 'Blue' else '105°C - 115°C (220°F - 240°F) for 2 to 3 hours'
        return {
            'status_level': 'danger',
            'status_title': 'Exhausted / Saturated (Regeneration Required)',
            'description': f'Moisture capacity fully reached. The {gel_type} silica gel can no longer absorb humidity from the atmosphere.',
            'action': f'Regenerate by baking in an oven or desiccant dryer at {temp} until original color is restored.'
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/samples', methods=['GET'])
def get_samples():
    sample_list = []
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for s in SAMPLE_IMAGES:
        abs_p = os.path.join(base_dir, s['rel_path'])
        if os.path.exists(abs_p):
            sample_list.append({
                'id': s['id'],
                'label': s['label'],
                'tag': s['tag'],
                'path': s['rel_path'],
                'url': f"/api/sample_img/{s['id']}"
            })
    return jsonify({'samples': sample_list})

@app.route('/api/sample_img/<sample_id>')
def serve_sample(sample_id):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for s in SAMPLE_IMAGES:
        if s['id'] == sample_id:
            abs_p = os.path.join(base_dir, s['rel_path'])
            if os.path.exists(abs_p):
                return send_file(abs_p, mimetype='image/jpeg')
    abort(404)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        pil_image = None
        
        # Check if file upload
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            pil_image = Image.open(file.stream).convert('RGB')
            
        # Check if base64 data
        elif request.is_json and 'image_base64' in request.json:
            b64_str = request.json['image_base64']
            if ',' in b64_str:
                b64_str = b64_str.split(',', 1)[1]
            img_bytes = base64.b64decode(b64_str)
            pil_image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            
        # Check if sample id / path
        elif request.is_json and 'sample_id' in request.json:
            sample_id = request.json['sample_id']
            base_dir = os.path.dirname(os.path.abspath(__file__))
            for s in SAMPLE_IMAGES:
                if s['id'] == sample_id:
                    abs_p = os.path.join(base_dir, s['rel_path'])
                    if os.path.exists(abs_p):
                        pil_image = Image.open(abs_p).convert('RGB')
                        break
                        
        if pil_image is None:
            return jsonify({'error': 'No valid image provided. Please upload an image file or choose a sample.'}), 400

        # Perform inference
        result = predict_silica_gel(pil_image, model=MODEL, device=DEVICE, show_plot=False)
        
        # Add recommendation
        rec = get_recommendation(result['gel_type'], result['condition'])
        result['recommendation'] = rec
        result['hardware'] = str(DEVICE).upper()
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Silica Gel Inference Web UI on http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)

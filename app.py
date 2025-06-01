import io
import base64
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from flask import Flask, request, render_template, redirect, url_for, jsonify

from models.cnn_model import SimpleCNN
from utils import GradCAM  # Make sure utils.py defines this!
import cv2

app = Flask(__name__)

# Device and model setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4
class_names = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']

model = SimpleCNN(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load('models/cnn_model.pth', map_location=DEVICE))
model.to(DEVICE)
model.eval()

# GradCAM setup - adjust target_layer as needed
target_layer = model.conv3
gradcam = GradCAM(model, target_layer)

# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# In-memory prediction history, newest first
history = []

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return transform(image).unsqueeze(0)

def create_heatmap_overlay(orig_img, heatmap):
    orig_img_np = np.array(orig_img.resize((224, 224)))

    # Convert heatmap to color
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Blend heatmap and original image
    overlay = cv2.addWeighted(orig_img_np, 0.6, heatmap_color, 0.4, 0)
    return overlay

def to_data_url(img_array):
    pil_img = Image.fromarray(img_array)
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def img_to_data_url(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            # No file uploaded, redirect back to GET
            return redirect(url_for('index'))

        img_bytes = file.read()
        try:
            # Preprocess and predict
            input_tensor = preprocess_image(img_bytes).to(DEVICE)
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            pred_label = class_names[pred_idx]
            confidence = probs[0, pred_idx].item()

            # Generate heatmap with GradCAM
            heatmap = gradcam.generate_heatmap(input_tensor, class_idx=pred_idx)

            # Load original image and overlay
            orig_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            overlay_img = create_heatmap_overlay(orig_img, heatmap)

            # Convert images to base64 data URLs for inline display
            orig_data_url = img_to_data_url(orig_img)
            overlay_data_url = to_data_url(overlay_img)

            # Add result to history (newest first)
            history.insert(0, {
                'orig_img': orig_data_url,
                'heatmap_img': overlay_data_url,
                'label': pred_label,
                'confidence': confidence
            })

            # Limit history size to last 10 entries
            if len(history) > 10:
                history.pop()

        except Exception as e:
            print("Error processing image:", e)
            # Optionally flash error message here

        # Redirect after POST to avoid form resubmission and duplicate processing
        return redirect(url_for('index'))

    # GET request renders page with current history
    return render_template('index.html', history=history)

@app.route('/delete_history/<int:index>', methods=['POST'])
def delete_history_item(index):
    """Delete a specific history item by index"""
    try:
        if 0 <= index < len(history):
            history.pop(index)
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'Invalid index'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/clear_history', methods=['POST'])
def clear_all_history():
    """Clear all history items"""
    try:
        global history
        history = []
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
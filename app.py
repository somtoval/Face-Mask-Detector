from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import os
import traceback

app = Flask(__name__)

# Load YOLOv8 model
try:
    model = YOLO('./best.pt')
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

def process_image(image_bytes):
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Failed to decode image")

        # Make a copy of the image for drawing
        img_draw = img.copy()
        
        # Run inference with YOLOv8
        results = model(img, verbose=False)
        
        # Initialize counters
        total_faces = 0
        with_mask = 0
        
        # Process results
        for result in results:
            boxes = result.boxes
            total_faces = len(boxes)
            
            for box in boxes:
                # Get box coordinates
                coords = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, coords)
                
                # Get class and confidence
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                
                # Update mask counter
                if class_id == 0:  # Assuming 0 is 'with_mask'
                    with_mask += 1
                
                # Draw rectangle and label
                color = (0, 255, 0) if class_id == 0 else (0, 0, 255)
                label = f"{'Mask' if class_id == 0 else 'No Mask'} {confidence:.2f}"
                
                cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img_draw, label, (x1, y1 - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Convert processed image to base64
        _, buffer = cv2.imencode('.jpg', img_draw)
        if buffer is None:
            raise ValueError("Failed to encode processed image")
            
        processed_image = base64.b64encode(buffer).decode('utf-8')
        
        return {
            'processed_image': processed_image,
            'total_faces': total_faces,
            'with_mask': with_mask,
            'without_mask': total_faces - with_mask
        }
        
    except Exception as e:
        print(f"Error in process_image: {str(e)}")
        print(traceback.format_exc())
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_mask():
    if model is None:
        return jsonify({'error': 'Model not loaded properly'}), 500
        
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        image_file = request.files['image']
        image_bytes = image_file.read()
        
        if not image_bytes:
            return jsonify({'error': 'Empty image provided'}), 400
            
        result = process_image(image_bytes)
        return jsonify(result)
    
    except Exception as e:
        print(f"Error in detect_mask: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Ensure the template directory exists
    template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
    if not os.path.exists(template_dir):
        os.makedirs(template_dir)
        
    # Check if frontend file exists
    index_path = os.path.join(template_dir, 'index.html')
    if not os.path.exists(index_path):
        print(f"Please create the file: {index_path}")
        print("Copy the frontend HTML code into this file.")
        exit(1)
    
    app.run(debug=True, port=5000)
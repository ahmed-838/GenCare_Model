from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
import os
import json
from model_inference import get_prediction, TARGET_CONDITIONS

app = Flask(__name__)
# Enable CORS for all routes
CORS(app)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit upload size to 16 MB
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create upload folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint to check if the API is running"""
    return jsonify({"status": "healthy"}), 200

@app.route('/api/conditions', methods=['GET'])
def get_conditions():
    """Endpoint to return the list of target conditions"""
    return jsonify({
        "success": True,
        "target_conditions": TARGET_CONDITIONS
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    API endpoint for image prediction
    
    Request: multipart/form-data with an image file
    Response: JSON with prediction results
    """
    # Check if request contains a file
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded", "success": False}), 400
        
    file = request.files['file']
    
    # Check if file is selected
    if file.filename == '':
        return jsonify({"error": "No file selected", "success": False}), 400
    
    # Check if file type is allowed
    if not allowed_file(file.filename):
        return jsonify({
            "error": f"File type not allowed. Allowed types: {', '.join(app.config['ALLOWED_EXTENSIONS'])}",
            "success": False
        }), 400
        
    try:
        # Save file temporarily
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process the image with the model
        result = get_prediction(file_path)
        
        # Return the results
        return jsonify({
            "success": True,
            "filename": filename,
            "results": result,
            "diagnosis_message": result.get("diagnosis_message", "")
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
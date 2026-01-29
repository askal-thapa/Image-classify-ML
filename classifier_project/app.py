import os
import json
import time
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from predict import load_inference_model, predict_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['TEST_CASES_FOLDER'] = 'static/test_cases_served'
app.config['DATA_FILE'] = 'static/data/test_results.json'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Ensure directories exist
# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.dirname(app.config['DATA_FILE']), exist_ok=True)

# Initialize JSON if not exists
if not os.path.exists(app.config['DATA_FILE']):
    with open(app.config['DATA_FILE'], 'w') as f:
        json.dump([], f)

print("Loading model for web app...")
model = load_inference_model()
print("Model loaded.")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'heic', 'heif', 'avif', 'tiff', 'tif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/details')
def details():
    return render_template('details.html')

@app.route('/deep-process')
def deep_process():
    return render_template('deep_process.html')

@app.route('/test-examples')
def test_examples():
    return render_template('test_examples.html')

@app.route('/references')
def references():
    # Read classification report
    report_path = os.path.join('static', 'metrics', 'classification_report.txt')
    report_content = "Report not found."
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            report_content = f.read()
            
    return render_template('references.html', report_content=report_content)

@app.route('/apidocs')
def apidocs():
    return render_template('apidocs.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Predict
        try:
            label, confidence = predict_image(model, filepath)
            
            # predict_image returns confidence of the predicted label.
            # script.js expects data.raw_score to be probability of Real?
            # Let's check script.js:
            # probReal = data.raw_score;
            # probAI = 1 - data.raw_score;
            # label: 'Real' or 'AI Generated'
            
            # My predict.py logic:
            # score = prediction[0][0] (sigmoid output)
            # if score >= 0.5: label=Real, conf=score (so score is prob of Real)
            # else: label=AI, conf=1-score (so score is prob of Real)
            
            # We need raw sigmoid score for the chart
            # I need to modify predict_image to return raw score or calculate it back
            # Or just hack it here. 
            # In predict.py:
            # if score >= 0.5: label=CLASS_NAMES[1] (Real)
            # else: label=CLASS_NAMES[0] (AI)
            
            # Let's assume predict_image logic is:
            # AI=0, Real=1.
            # If label is Real, confidence is score.
            # If label is AI, confidence is 1-score.
            
            if label == 'Real':
                raw_score = confidence
            else:
                raw_score = 1.0 - confidence
                
            response = {
                'label': 'AI Generated' if label == 'AI' else 'Real',
                'confidence': float(confidence),
                'raw_score': float(raw_score),
                'filename': filename
            }
            return jsonify(response)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
            
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/save-test-case', methods=['POST'])
def save_test_case():
    if 'image' not in request.files or 'ground_truth' not in request.form:
        return jsonify({'error': 'Missing data'}), 400
        
    file = request.files['image']
    ground_truth = request.form['ground_truth']
    
    if file and allowed_file(file.filename):
        # Determine target folder based on Ground Truth
        # ground_truth is 'Real' or 'AI Generated' (or 'AI')
        # We want folders: static/uploads/Real and static/uploads/AI
        
        target_subfolder = 'Real' if ground_truth == 'Real' else 'AI'
        target_dir = os.path.join('static', 'uploads', target_subfolder)
        os.makedirs(target_dir, exist_ok=True)

        # Generate unique filename
        timestamp = str(int(time.time()))
        filename = secure_filename(f"{timestamp}_{file.filename}")
        filepath = os.path.join(target_dir, filename)
        
        # Save file directly to the structured folder
        file.save(filepath)
        
        # Cleanup: Remove the original temporary file from uploads if it exists
        # Reconstruct the temp path using the original filename
        temp_filename = secure_filename(file.filename)
        temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        if os.path.exists(temp_filepath):
            try:
                os.remove(temp_filepath)
                print(f"Removed temp file: {temp_filepath}")
            except Exception as e:
                print(f"Error removing temp file: {e}")
        
        # Predict again to get stats (or we could pass them from frontend but re-predict is safer)
        label, confidence = predict_image(model, filepath)
        pred_label = 'AI Generated' if label == 'AI' else 'Real'
        
        # Record entry
        # Image path for frontend needs to be relative url
        web_path = f"/static/uploads/{target_subfolder}/{filename}"
        
        entry = {
            'filename': filename,
            'original_label': ground_truth,
            'predicted_label': pred_label,
            'confidence': float(confidence),
            'raw_score': float(confidence) if label == 'Real' else float(1.0 - confidence),
            'image_path': web_path,
            'timestamp': time.time()
        }
        
        # Append to JSON
        try:
            with open(app.config['DATA_FILE'], 'r') as f:
                data = json.load(f)
        except:
            data = []
            
        data.append(entry)
        
        with open(app.config['DATA_FILE'], 'w') as f:
            json.dump(data, f, indent=4)
            
        return jsonify({'success': True})

    return jsonify({'error': 'Save failed'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8000)

# app.py - Final Version

import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend for server environments
import matplotlib.pyplot as plt

# --- App Initialization ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/'
app.config['SECRET_KEY'] = 'your-super-secret-key-for-project' # It's good practice to have a secret key

# --- Configuration ---
IMG_SIZE = (128, 128)

# *** IMPORTANT: The Threshold Dictionary ***
# This is where you will store the calibrated threshold for each product.
# Add a new entry here after you test and find the best value for each model.
THRESHOLDS = {
    'bottle': 0.0015,
    'cable': 0.0045,  # <-- Our newly calibrated value!
    'capsule': 0.0005,  # <-- Our newly calibrated value!
    'carpet': 0.0020,  # <-- Our newly calibrated value!
    'grid': 0.0020,  # <-- Our newly calibrated value!
    'hazelnut': 0.0015,
    'leather': 0.001,  # <-- Our newly calibrated value!
    'metal_nut': 0.0020,  # <-- Our newly calibrated value!
    'pill': 0.0009,  # <-- Our newly calibrated value!
    'screw': 0.0006,  # <-- Our newly calibrated value!
    'tile': 0.0020,  # <-- Our newly calibrated value!
    'toothbrush': 0.0020,  # <-- Our newly calibrated value!
    'transistor': 0.0018,  # <-- Our newly calibrated value!
    'wood': 0.0020,  # <-- Our newly calibrated value!
    'zipper': 0.0020,  # <-- Our newly calibrated value!
    'default': 0.0030    # A fallback threshold for any model not in this list
}

# --- Model Loading Cache ---
# This dictionary stores loaded models in memory to improve performance.
models_cache = {}

def get_model(product_type):
    """Dynamically loads a model into memory and caches it."""
    if product_type in models_cache:
        return models_cache[product_type]
    
    model_path = f"models/{product_type}_model.h5"
    if not os.path.exists(model_path):
        return None # Model file doesn't exist
        
    print(f"INFO: Loading model for '{product_type}' from disk...")
    model = tf.keras.models.load_model(model_path, compile=False)
    models_cache[product_type] = model
    print("INFO: Model loaded and cached.")
    return model

# --- Helper Functions ---
def preprocess_image(image_path):
    """Loads and preprocesses an image from a file path."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype('float32') / 255.0
    img = np.reshape(img, (1, IMG_SIZE[0], IMG_SIZE[1], 1))
    return img

def create_heatmap_report(original_img, reconstructed_img, original_filename):
    """Calculates the difference and saves a heatmap report image."""
    original = np.squeeze(original_img)
    reconstructed = np.squeeze(reconstructed_img)
    diff = np.abs(original - reconstructed)
    
    fig, ax = plt.subplots()
    im = ax.imshow(diff, cmap='hot')
    fig.colorbar(im)
    ax.axis('off') # Hide axes for a cleaner look
    
    # Create a unique filename for the report
    report_filename = f"report_{original_filename}"
    report_path = os.path.join(app.config['UPLOAD_FOLDER'], report_filename)
    fig.savefig(report_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig) # Prevent memory leaks by closing the plot
    return report_filename

# --- Web App Routes ---
@app.route('/', methods=['GET', 'POST'])
def home():
    """Handles the main page, file uploads, and analysis."""
    # Automatically find which models have been trained and are available in the 'models' folder
    try:
        available_products = [f.replace('_model.h5', '') for f in os.listdir('models') if f.endswith('.h5')]
    except FileNotFoundError:
        available_products = []
    
    if request.method == 'POST':
        product_type = request.form.get('product_type')
        file = request.files.get('file')

        # Basic validation
        if not all([product_type, file, file.filename]):
            return render_template('index.html', available_products=available_products, error="Error: Please select a product type and upload a file.")

        # Load the correct model based on user selection
        model = get_model(product_type)
        if model is None:
            error_msg = f"Error: Model for '{product_type}' not found. Please train it first using the training notebook."
            return render_template('index.html', available_products=available_products, error=error_msg)
            
        filename = secure_filename(file.filename)
        original_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(original_filepath)
        
        # --- Perform Analysis ---
        test_image = preprocess_image(original_filepath)
        reconstructed_image = model.predict(test_image)
        error = np.mean(np.square(test_image - reconstructed_image))
        
        # Get the specific threshold for the selected product, or use the default
        error_threshold = THRESHOLDS.get(product_type, THRESHOLDS['default'])
        
        # Make the final decision
        is_defective = error > error_threshold
        verdict = "DEFECTIVE" if is_defective else "Not Defective"
        
        # Generate the visual report
        report_img_filename = create_heatmap_report(test_image, reconstructed_image, filename)

        # Display the results page
        return render_template('result.html', 
                               verdict=verdict,
                               defective=is_defective,
                               error_score=error,
                               threshold=error_threshold,
                               original_img=filename,
                               report_img=report_img_filename)
            
    return render_template('index.html', available_products=available_products)

# This is the standard entry point for running a Flask app
if __name__ == '__main__':
    # Make sure the necessary folders exist before starting
    os.makedirs('models', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    app.run(debug=True)














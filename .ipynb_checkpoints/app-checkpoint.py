{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "fb992bf9-1d04-45a1-b020-8144df9c3a6d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " * Serving Flask app '__main__'\n",
      " * Debug mode: on\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.\n",
      " * Running on http://127.0.0.1:5000\n",
      "Press CTRL+C to quit\n",
      " * Restarting with watchdog (windowsapi)\n"
     ]
    },
    {
     "ename": "SystemExit",
     "evalue": "1",
     "output_type": "error",
     "traceback": [
      "An exception has occurred, use %tb to see the full traceback.\n",
      "\u001b[1;31mSystemExit\u001b[0m\u001b[1;31m:\u001b[0m 1\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\user\\AppData\\Roaming\\Python\\Python312\\site-packages\\IPython\\core\\interactiveshell.py:3585: UserWarning: To exit: use 'exit', 'quit', or Ctrl-D.\n",
      "  warn(\"To exit: use 'exit', 'quit', or Ctrl-D.\", stacklevel=1)\n"
     ]
    }
   ],
   "source": [
    "# app.py\n",
    "\n",
    "import os\n",
    "import cv2\n",
    "import numpy as np\n",
    "import tensorflow as tf\n",
    "from flask import Flask, request, render_template, url_for\n",
    "from werkzeug.utils import secure_filename\n",
    "import matplotlib\n",
    "matplotlib.use('Agg') # Use a non-interactive backend for server environments\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# --- App Initialization ---\n",
    "app = Flask(__name__)\n",
    "app.config['UPLOAD_FOLDER'] = 'static/'\n",
    "app.config['SECRET_KEY'] = 'your-super-secret-key'\n",
    "\n",
    "# --- Configuration ---\n",
    "IMG_SIZE = (128, 128)\n",
    "# This general threshold works reasonably well. For a more fine-tuned system,\n",
    "# you might define a specific threshold for each model.\n",
    "ERROR_THRESHOLD = 0.0030\n",
    "\n",
    "# --- Model Loading Cache ---\n",
    "# This dictionary will store loaded models in memory to avoid reloading from disk on every request.\n",
    "models_cache = {}\n",
    "\n",
    "def get_model(product_type):\n",
    "    \"\"\"Dynamically loads a model into memory and caches it.\"\"\"\n",
    "    if product_type in models_cache:\n",
    "        return models_cache[product_type]\n",
    "    \n",
    "    model_path = f\"models/{product_type}_model.h5\"\n",
    "    if not os.path.exists(model_path):\n",
    "        return None # Model file doesn't exist\n",
    "        \n",
    "    print(f\"INFO: Loading model for '{product_type}' from disk...\")\n",
    "    model = tf.keras.models.load_model(model_path, compile=False)\n",
    "    models_cache[product_type] = model\n",
    "    print(\"INFO: Model loaded and cached.\")\n",
    "    return model\n",
    "\n",
    "# --- Helper Functions ---\n",
    "def preprocess_image(image_path):\n",
    "    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)\n",
    "    img = cv2.resize(img, IMG_SIZE)\n",
    "    img = img.astype('float32') / 255.0\n",
    "    img = np.reshape(img, (1, IMG_SIZE[0], IMG_SIZE[1], 1))\n",
    "    return img\n",
    "\n",
    "def create_heatmap_report(original_img, reconstructed_img, original_filename):\n",
    "    \"\"\"Calculates the difference and saves a heatmap report image.\"\"\"\n",
    "    original = np.squeeze(original_img)\n",
    "    reconstructed = np.squeeze(reconstructed_img)\n",
    "    diff = np.abs(original - reconstructed)\n",
    "    \n",
    "    fig, ax = plt.subplots()\n",
    "    im = ax.imshow(diff, cmap='hot')\n",
    "    fig.colorbar(im)\n",
    "    ax.axis('off') # Hide axes for a cleaner look\n",
    "    \n",
    "    report_filename = f\"report_{original_filename}\"\n",
    "    report_path = os.path.join(app.config['UPLOAD_FOLDER'], report_filename)\n",
    "    fig.savefig(report_path, bbox_inches='tight', pad_inches=0)\n",
    "    plt.close(fig) # Prevent memory leaks\n",
    "    return report_filename\n",
    "\n",
    "# --- Web App Routes ---\n",
    "@app.route('/', methods=['GET', 'POST'])\n",
    "def home():\n",
    "    # Automatically find which models have been trained and are available\n",
    "    available_products = [f.replace('_model.h5', '') for f in os.listdir('models') if f.endswith('.h5')]\n",
    "    \n",
    "    if request.method == 'POST':\n",
    "        product_type = request.form.get('product_type')\n",
    "        file = request.files.get('file')\n",
    "\n",
    "        if not all([product_type, file, file.filename]):\n",
    "            return render_template('index.html', available_products=available_products, error=\"Error: Please select a product type and upload a file.\")\n",
    "\n",
    "        model = get_model(product_type)\n",
    "        if model is None:\n",
    "            error_msg = f\"Error: Model for '{product_type}' not found. Please train it using the train_multi.py script.\"\n",
    "            return render_template('index.html', available_products=available_products, error=error_msg)\n",
    "            \n",
    "        filename = secure_filename(file.filename)\n",
    "        original_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)\n",
    "        file.save(original_filepath)\n",
    "        \n",
    "        test_image = preprocess_image(original_filepath)\n",
    "        reconstructed_image = model.predict(test_image)\n",
    "        error = np.mean(np.square(test_image - reconstructed_image))\n",
    "        is_defective = error > ERROR_THRESHOLD\n",
    "        verdict = \"DEFECTIVE\" if is_defective else \"Not Defective\"\n",
    "        report_img_filename = create_heatmap_report(test_image, reconstructed_image, filename)\n",
    "\n",
    "        return render_template('result.html', \n",
    "                               verdict=verdict,\n",
    "                               defective=is_defective,\n",
    "                               error_score=error,\n",
    "                               threshold=ERROR_THRESHOLD,\n",
    "                               original_img=filename,\n",
    "                               report_img=report_img_filename)\n",
    "            \n",
    "    return render_template('index.html', available_products=available_products)\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    # Make sure the static folder exists\n",
    "    os.makedirs('static', exist_ok=True)\n",
    "    app.run(debug=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d9665343-1df3-4e96-9cc5-5b52c5255eae",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

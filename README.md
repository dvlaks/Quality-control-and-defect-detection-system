# AI-Powered Quality Control & Defect Detection System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-Flask-black.svg)](https://flask.palletsprojects.com/)
[![Deep Learning](https://img.shields.io/badge/Library-TensorFlow-orange.svg)](https://www.tensorflow.org/)

An end-to-end web application for automated visual quality control in manufacturing. This project leverages unsupervised deep learning to identify and localize defects on various industrial products, providing a scalable solution to overcome the limitations of manual inspection.


## 🎯 Project Overview

In modern manufacturing, maintaining high product quality is a strategic imperative. Traditional manual inspection is often slow, subjective, and expensive, making it a bottleneck in high-volume production. This project, developed as part of an Industrial and Production Engineering curriculum, addresses this challenge by creating an intelligent system for automated defect detection.

The core of the system is an **unsupervised anomaly detection** model. Using **Convolutional Autoencoders (CAEs)**, the system learns the "normal" appearance of a product from defect-free examples. When a new product is inspected, the system flags any deviation from this learned norm as a potential defect, providing both a verdict and a visual heatmap to pinpoint the anomaly's location.

## ✨ Key Features

-   **Web-Based Interface:** A user-friendly UI built with Flask allows easy access for quality control personnel to upload images and receive instant feedback.
-   **Unsupervised Learning:** The system trains only on "good" product images, making it highly practical for industrial settings where defect data is scarce and unpredictable.
-   **Multi-Product Scalability:** The architecture supports multiple product lines. New product models can be trained and added without changing the core application logic.
-   **Visual Defect Localization:** Generates a **heatmap** for every defective item, providing clear visual evidence of the anomaly's location and shape, which is crucial for root cause analysis.
-   **Calibrated Sensitivity:** Features a model-specific thresholding mechanism, allowing the detection sensitivity to be fine-tuned for each product category to balance precision and recall.

## 🛠️ Technology Stack

-   **Backend:** Python, Flask
-   **Machine Learning:** TensorFlow, Keras
--   **Image Processing:** OpenCV, NumPy
-   **Data & Training:** Jupyter Notebook, Matplotlib
-   **Frontend:** HTML, CSS

## 📂 System Architecture

The system is designed as a modular client-server application:

1.  **Frontend (Client):** The user interacts with a simple web page to select a product type and upload an image.
2.  **Backend (Flask Server):**
    -   Handles the image upload.
    -   Dynamically loads the appropriate pre-trained CAE model for the selected product.
    -   Preprocesses the image to match the model's input requirements.
3.  **ML Inference Engine (TensorFlow/Keras):**
    -   The model processes the image and attempts to reconstruct it.
    -   It calculates the **reconstruction error** (Mean Squared Error) between the original and reconstructed image.
4.  **Results Generation:**
    -   The error score is compared against a calibrated threshold to classify the item as "Defective" or "Not Defective."
    -   A heatmap is generated from the pixel-wise difference, and the results are sent back to the user.

## 📊 Dataset

This project uses the **MVTec Anomaly Detection (MVTec AD) dataset**. It is a comprehensive, real-world dataset designed for unsupervised anomaly detection in industrial inspection.

-   **Content:** 15 categories (5 textures, 10 objects) like leather, wood, bottle, screw, etc.
-   **Structure:** The training set contains only defect-free images, perfectly aligning with our unsupervised approach. The test set contains both good and defective images with over 70 different types of anomalies.
-   **Credit:** Bergmann, P., et al. (2019, 2021). *The MVTec Anomaly Detection Dataset*.

## 🚀 Getting Started

Follow these steps to set up and run the project on your local machine.

### Prerequisites

-   Git
-   Python 3.7+ and Pip
-   A virtual environment tool (e.g., `venv`) is recommended.

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[Your-Username]/[Your-Repository-Name].git
    cd [Your-Repository-Name]
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    *(First, create the requirements.txt file with `pip freeze > requirements.txt` after installing all packages in your virtual environment)*
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the MVTec AD Dataset:**
    -   Download the dataset from the [MVTec AD Website](https://www.mvtec.com/company/research/datasets/mvtec-ad).
    -   Extract it and place the contents into a `datasets/` folder in the root of the project directory. Your folder structure should look like this:
        ```
        project-root/
        ├── datasets/
        │   ├── bottle/
        │   ├── cable/
        │   └── ... (other product folders)
        ├── models/
        ├── static/
        └── app.py
        ```

### Usage

The project has two main parts: training the models and running the web application.

1.  **Train the Models:**
    -   Open and run the `Model_Training.ipynb` Jupyter Notebook.
    -   Set the `PRODUCT_NAME` variable in the notebook to the category you want to train (e.g., `'bottle'`).
    -   Run all cells. The trained model will be saved as `models/{PRODUCT_NAME}_model.h5`.
    -   Repeat this process for all product categories you want to include in your application.

2.  **Run the Web Application:**
    -   Execute the Flask app from your terminal:
        ```bash
        python app.py
        ```
    -   Open your web browser and navigate to `http://127.0.0.1:5000`.
    -   Select a product type from the dropdown, upload an image, and see the results!

## 🔬 The Calibration Process

A key finding of this project was that a single, universal error threshold is ineffective. Each product and model has a unique error profile. To ensure accuracy, you must calibrate the threshold for each model.

This is done in the `THRESHOLDS` dictionary within `app.py`:

```python
THRESHOLDS = {
    'bottle': 0.0015,
    'leather': 0.0020,
    'cable': 0.0035,
    # Add other calibrated thresholds here
    'default': 0.0030
}

# LucidLens: Real-Time Image Captioning Platform

## Overview
LucidLens is a project that provides real-time image captioning using a pre-trained VGG16 model and a custom-trained caption generation model. It allows users to upload images, capture photos via webcam, or process video input to generate descriptive captions, which are then converted to speech.

## Features
* **Image Captioning:** Generates descriptive captions for images using a deep learning model.
* **Real-Time Processing:** Provides near real-time caption generation for uploaded images, captured photos, and video input.
* **Text-to-Speech:** Converts generated captions into spoken words using Google Text-to-Speech (gTTS).
* **Multiple Input Options:** Supports image uploads, direct photo capture via webcam, and video stream processing.
* **User-Friendly Interface:** Utilizes Streamlit for an intuitive and interactive web application.

## Technologies Used
* **Python:** Primary programming language.
* **TensorFlow/Keras:** Deep learning framework for image feature extraction and caption generation.
* **VGG16:** Pre-trained convolutional neural network for image feature extraction.
* **gTTS (Google Text-to-Speech):** Converts text captions to speech.
* **Pygame:** Plays the generated audio.
* **Streamlit:** Creates the web application interface.
* **OpenCV:** Used for processing video frames and capturing images from the webcam.
* **Streamlit-webrtc:** Enables real-time video streaming from the webcam within the Streamlit app.
* **PIL (Pillow):** Python Imaging Library for image processing.
* **NumPy:** For numerical operations, especially handling image data.
* **gdown:** For downloading necessary files.

## Setup and Installation

### 1. Clone the Repository
```bash
git clone https://github.com/govindvgd/LucidLens.git
cd LucidLens
```

### 2. Install Dependencies
Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Linux/macOS
venv\Scripts\activate  # On Windows
```

Install the required Python packages:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Run the Streamlit Application
```bash
streamlit run main.py
```
This will start the Streamlit server and open the application in your web browser.

### 2. Interact with the Application
* **Upload an image:** Click "Browse files" to upload an image from your computer. The application will display the image and generate a caption. The caption will also be spoken aloud.
* **Capture a photo:** Select the "Capture a photo" option. Your webcam will activate, and you'll see a live video feed. Press "c" on your keyboard to capture a frame. The application will display the captured image and generate a caption.
* **Video input:** Select the "Video input" option to process video stream from your webcam and generate captions continuously.

## Model Training

### 1. Image Feature Extraction Model
* **Architecture:** The VGG16 model is used as the base for feature extraction. The fully connected layers at the top of the VGG16 model are removed, and the output of the last convolutional layer (`block5_pool`) is used as the feature vector for each image.
* **Implementation:** The code uses `tensorflow.keras.applications.VGG16` to load the pre-trained VGG16 model. A new Keras `Model` is created, taking the input of the VGG16 model and outputting the layer before the fully connected layers.
* **Pre-processing:** Images are pre-processed using `tensorflow.keras.applications.vgg16.preprocess_input` to normalize pixel values, ensuring compatibility with the pre-trained VGG16 model.

### 2. Feature Extraction from Dataset
* **Dataset:** The code assumes the existence of a directory containing the images. The directory should be replaced with the actual path to your image dataset.
* **Batch Processing:** To handle large datasets efficiently, images are processed in batches.
* **Feature Extraction Loop:** The code iterates through the images, loads them, pre-processes them, and extracts features using the VGG16 model.

### 3. Saving Image Features
* **Pickle Format:** The extracted image features are saved to a pickle file (`image_features.pkl`).
* **Loading Features:** The code includes functionality to load the extracted features from the pickle file.

### 4. Caption Mapping Creation
* **Caption File:** The code reads caption data from a text file.
* **Mapping Dictionary:** A dictionary called `mapping` is created to store the mapping between image IDs and captions.
* **Data Cleaning:** The code performs basic cleaning of the caption text.
* **Tokenization and Vocabulary Creation:** Tokenization creates a vocabulary for the words in the captions.

## Acknowledgements

*   This project utilizes the VGG16 model pre-trained on ImageNet.
*   The Streamlit and streamlit-webrtc libraries are essential for the user interface and webcam integration.
*   The gTTS library enables the text-to-speech functionality.


import os
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pygame
from gtts import gTTS
from PIL import Image
import cv2
from streamlit_webrtc import VideoTransformerBase
import streamlit as st
import gdown

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model_path = "fastest_model.keras"
        
        # Check if the model file exists, otherwise download it
        if not os.path.exists(self.model_path):
            url = 'https://drive.google.com/uc?id=19pACqei0GBVA4DJsNWJq025oxMUsooqT'  # Corrected download link
            output = self.model_path
            gdown.download(url, output, quiet=False)
        
        # Check again if the file exists after the download
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}. Download failed or file is missing.")
        
        # Now load the model
        self.model = tf.keras.models.load_model(self.model_path)
        self.mapping_path = "mapping.pkl"  # Update with your mapping path
        self.mapping = self.load_mapping(self.mapping_path)
        self.tokenizer = self.create_tokenizer(self.mapping)

    def idx_to_word(self, integer, tokenizer):
        for word, index in tokenizer.word_index.items():
            if index == integer:
                return word
        return None

    def load_mapping(self, file_path):
        with open(file_path, 'rb') as f:
            mapping = pickle.load(f)
        return mapping

    def create_tokenizer(self, mapping):
        all_captions = [caption for captions in mapping.values() for caption in captions]
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(all_captions)
        return tokenizer

    def load_image_features(self, image_path):
        base_model = VGG16()
        vgg_model = Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)
        features = vgg_model.predict(np.expand_dims(image, axis=0))[0, ...]
        return features

    def generate_caption(self, image_features):
        caption = 'startseq'
        max_length = 80
        for _ in range(max_length):
            sequence = self.tokenizer.texts_to_sequences([caption])[0]
            sequence = pad_sequences([sequence], maxlen=max_length)
            yhat = self.model.predict([np.expand_dims(image_features, axis=0), sequence], verbose=0)
            yhat = np.argmax(yhat)
            word = self.idx_to_word(yhat, self.tokenizer)
            if word is None or word == 'endseq':
                break
            caption += " " + word
            caption = caption.strip()  # Use strip() to remove leading/trailing whitespace

        # Remove the "startseq" token from the final caption
        caption = caption.replace('startseq', '').strip()
        return caption

    def generate_audio(self, caption, language='en', output_dir='audio_files', output_file='generated_audio1.mp3'):
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_file)
        tts = gTTS(text=caption, lang=language, slow=False)
        tts.save(output_path)
        pygame.mixer.init()
        pygame.mixer.music.load(output_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            continue

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        cv2.imwrite("temp_image.jpg", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image_features = self.load_image_features("temp_image.jpg")
        caption = self.generate_caption(image_features)
        self.generate_audio(caption)
        return image


def show_uploaded_image(image_file):
    img = Image.open(image_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)


def main(image_file, mapping_path):
    transformer = VideoTransformer()
    mapping = transformer.load_mapping(mapping_path)
    tokenizer = transformer.create_tokenizer(mapping)
    image_features = transformer.load_image_features(image_file)
    caption = transformer.generate_caption(image_features)
    transformer.generate_audio(caption)
    return caption

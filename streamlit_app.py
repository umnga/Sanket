"""
Nepali Sign Language Recognition - Minimal Streamlit App with Hand Detection
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, Reshape, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
import os
from PIL import Image, ImageDraw
import cv2
import mediapipe as mp

# Configure Streamlit page
st.set_page_config(
    page_title="Nepali Sign Language Recognition",
    layout="centered"
)

# LSTM Configuration
LSTM_CONFIG = {
    'img_size': (128, 128),
    'num_classes': 36,
    'sequence_length': 8,
    'lstm_units': 128,
    'dense_units': 256,
    'dropout_rate': 0.3,
}

# Class names for Nepali Sign Language
CLASS_NAMES = {
    0: "क (Ka)", 1: "ख (Kha)", 2: "ट (Ta)", 3: "ठ (Tha)", 4: "ड (Da)", 5: "ढ (Dha)",
    6: "ण (Na)", 7: "त (Ta)", 8: "थ (Tha)", 9: "द (Da)", 10: "ध (Dha)", 11: "न (Na)",
    12: "ग (Ga)", 13: "प (Pa)", 14: "फ (Pha)", 15: "ब (Ba)", 16: "भ (Bha)", 17: "म (Ma)",
    18: "य (Ya)", 19: "र (Ra)", 20: "ल (La)", 21: "व (Wa)", 22: "श (Sha)", 23: "घ (Gha)",
    24: "ष (Shha)", 25: "स (Sa)", 26: "ह (Ha)", 27: "क्ष (Ksha)", 28: "त्र (Tra)", 29: "ज्ञ (Gya)",
    30: "ङ (Nga)", 31: "च (Cha)", 32: "छ (Chha)", 33: "ज (Ja)", 34: "झ (Jha)", 35: "ञ (Nya)"
}

def create_hybrid_cnn_lstm_model():
    """Recreate the hybrid CNN-LSTM model architecture"""
    image_input = tf.keras.Input(shape=(*LSTM_CONFIG['img_size'], 3), name='input_1')
    
    base_cnn = MobileNetV2(weights='imagenet', include_top=False, 
                         input_shape=(*LSTM_CONFIG['img_size'], 3),
                         name='mobilenetv2_1.00_128')
    base_cnn.trainable = False
    
    cnn_features = base_cnn(image_input)
    shape = cnn_features.shape
    reshaped = Reshape((shape[1] * shape[2], shape[3]), name='reshape')(cnn_features)
    
    sequence_features = tf.keras.layers.Lambda(
        lambda x: x[:, :LSTM_CONFIG['sequence_length'], :],
        name='lambda'
    )(reshaped)
    
    lstm_out = LSTM(LSTM_CONFIG['lstm_units'], return_sequences=True, name='lstm')(sequence_features)
    lstm_out = LSTM(LSTM_CONFIG['lstm_units'] // 2, return_sequences=False, name='lstm_1')(lstm_out)
    
    global_features = GlobalAveragePooling2D(name='global_average_pooling2d')(cnn_features)
    combined = tf.keras.layers.concatenate([lstm_out, global_features], name='concatenate')
    
    x = Dense(LSTM_CONFIG['dense_units'], activation='relu', name='dense')(combined)
    x = BatchNormalization(name='batch_normalization')(x)
    x = Dropout(LSTM_CONFIG['dropout_rate'], name='dropout')(x)
    
    x = Dense(LSTM_CONFIG['dense_units'] // 2, activation='relu', name='dense_1')(x)
    x = BatchNormalization(name='batch_normalization_1')(x)
    x = Dropout(LSTM_CONFIG['dropout_rate'], name='dropout_1')(x)
    
    output = Dense(LSTM_CONFIG['num_classes'], activation='softmax', name='dense_2')(x)
    model = Model(inputs=image_input, outputs=output)
    return model

@st.cache_resource
def load_lstm_model():
    """Load the LSTM model"""
    model_paths = ['models/lstm_nsl_model.h5', 'models/lstm_nsl_checkpoint.h5']
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                model = tf.keras.models.load_model(model_path, compile=False)
                return model, model_path
            except Exception:
                try:
                    model = create_hybrid_cnn_lstm_model()
                    model.load_weights(model_path)
                    return model, model_path
                except Exception:
                    continue
    return None, None

def preprocess_image(image, use_hand_detection=True):
    """Preprocess image for model prediction with optional hand detection"""
    original_image = image.copy()
    detection_success = False
    annotated_image = image
    
    if use_hand_detection:
        # Try to detect and crop hand
        cropped_image, detection_success, annotated_image = detect_and_crop_hand(image)
        if detection_success:
            image = cropped_image
            st.success("Hand detected and cropped!")
        else:
            st.warning("No hand detected, using full image")
    
    # Standard preprocessing
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_resized = image.resize(LSTM_CONFIG['img_size'])
    img_array = np.array(image_resized).astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    
    return img_batch, image_resized, detection_success, annotated_image

def predict_image(model, image, use_hand_detection=True):
    """Make prediction on image with optional hand detection"""
    processed_image, display_image, detection_success, annotated_image = preprocess_image(image, use_hand_detection)
    predictions = model.predict(processed_image, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class])
    
    return {
        'predicted_class': predicted_class,
        'predicted_class_name': CLASS_NAMES.get(predicted_class, str(predicted_class)),
        'confidence': confidence,
        'display_image': display_image,
        'detection_success': detection_success,
        'annotated_image': annotated_image
    }

def display_result(result):
    """Display prediction result with hand detection visualization"""
    # Show images
    if result['detection_success']:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.image(result['annotated_image'], caption="Hand Detection", width=180)
        
        with col2:
            st.image(result['display_image'], caption="Cropped Hand", width=180)
        
        with col3:
            st.write("**Prediction:**")
            st.write(f"**{result['predicted_class_name']}**")
            st.write(f"**Confidence: {result['confidence']:.1%}**")
    else:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(result['display_image'], caption="Full Image", width=200)
        
        with col2:
            st.write("**Prediction:**")
            st.write(f"**{result['predicted_class_name']}**")
            st.write(f"**Confidence: {result['confidence']:.1%}**")

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def detect_and_crop_hand(image, show_detection=True):
    """
    Detect hand in image and crop to hand bounding box
    Returns: (cropped_image, detection_success, annotated_image)
    """
    # Convert PIL to cv2
    img_array = np.array(image)
    
    # Initialize hand detection
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    ) as hands:
        
        # MediaPipe expects RGB format, and PIL images are already RGB
        rgb_image = img_array
        
        # Process the image
        results = hands.process(rgb_image)
        
        # Create annotated image for display
        annotated_image = img_array.copy()
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]  # Get first hand
            
            # Get image dimensions
            h, w = img_array.shape[:2]
            
            # Calculate bounding box
            x_coords = [landmark.x * w for landmark in hand_landmarks.landmark]
            y_coords = [landmark.y * h for landmark in hand_landmarks.landmark]
            
            min_x, max_x = int(min(x_coords)), int(max(x_coords))
            min_y, max_y = int(min(y_coords)), int(max(y_coords))
            
            # Add padding around hand
            padding = 20
            min_x = max(0, min_x - padding)
            min_y = max(0, min_y - padding)
            max_x = min(w, max_x + padding)
            max_y = min(h, max_y + padding)
            
            # Crop the hand region
            cropped = img_array[min_y:max_y, min_x:max_x]
            
            if show_detection:
                # Draw bounding box on annotated image
                cv2.rectangle(annotated_image, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
                # Draw hand landmarks
                mp_drawing.draw_landmarks(annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Convert back to PIL
            cropped_pil = Image.fromarray(cropped)
            annotated_pil = Image.fromarray(annotated_image)
            
            return cropped_pil, True, annotated_pil
        else:
            # No hand detected, return original image
            return image, False, Image.fromarray(annotated_image)

def main():
    """Main Streamlit app"""
    st.title("Nepali Sign Language Recognition")
    st.write("LSTM-based recognition with camera and upload support")
    
    # Load model
    model, model_path = load_lstm_model()
    if model is None:
        st.error("Could not load model. Please check if model files exist.")
        return
    
    st.success(f"Model loaded from: {model_path}")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        options=["Upload Image", "Take Photo with Camera"],
        horizontal=True
    )
    
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader(
            "Choose an image of a sign language gesture",
            type=['png', 'jpg', 'jpeg']
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            # Detect and crop hand
            cropped_image, hand_detected, annotated_image = detect_and_crop_hand(image)
            if hand_detected:
                st.image(annotated_image, caption="Hand Detection", use_column_width=True)
            else:
                st.warning("No hand detected in the image.")
            
            with st.spinner("Processing..."):
                result = predict_image(model, cropped_image)
            display_result(result)
    
    elif input_method == "Take Photo with Camera":
        camera_image = st.camera_input("Take a photo of your sign language gesture")
        
        if camera_image is not None:
            image = Image.open(camera_image)
            # Detect and crop hand
            cropped_image, hand_detected, annotated_image = detect_and_crop_hand(image)
            if hand_detected:
                st.image(annotated_image, caption="Hand Detection", use_column_width=True)
            else:
                st.warning("No hand detected in the image.")
            
            with st.spinner("Processing..."):
                result = predict_image(model, cropped_image)
            display_result(result)

if __name__ == "__main__":
    main()

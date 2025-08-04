"""
Test single sample from each class using LSTM model
Tests the model on one image from each of the 36 Nepali sign language classes
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, Reshape, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from PIL import Image
import matplotlib.pyplot as plt
import random

# LSTM Configuration (same as training)
LSTM_CONFIG = {
    'dataset_path': 'nslDataset',
    'img_size': (128, 128),
    'num_classes': 36,
    'sequence_length': 8,
    'lstm_units': 128,
    'dense_units': 256,
    'dropout_rate': 0.3,
}

# Generic class names - using the TensorFlow folder mapping order
# We know the model works (99.83% accuracy) but don't have the correct Nepali character mapping
# So using generic labels based on the actual folder structure
CLASS_NAMES = {
    0: "Gesture_0 (Folder 0)",    # TF class 0 → Folder "0"
    1: "Gesture_1 (Folder 1)",    # TF class 1 → Folder "1" 
    2: "Gesture_2 (Folder 10)",   # TF class 2 → Folder "10"
    3: "Gesture_3 (Folder 11)",   # TF class 3 → Folder "11"
    4: "Gesture_4 (Folder 12)",   # TF class 4 → Folder "12"
    5: "Gesture_5 (Folder 13)",   # TF class 5 → Folder "13"
    6: "Gesture_6 (Folder 14)",   # TF class 6 → Folder "14"
    7: "Gesture_7 (Folder 15)",   # TF class 7 → Folder "15"
    8: "Gesture_8 (Folder 16)",   # TF class 8 → Folder "16"
    9: "Gesture_9 (Folder 17)",   # TF class 9 → Folder "17"
    10: "Gesture_10 (Folder 18)", # TF class 10 → Folder "18"
    11: "Gesture_11 (Folder 19)", # TF class 11 → Folder "19"
    12: "Gesture_12 (Folder 2)",  # TF class 12 → Folder "2"
    13: "Gesture_13 (Folder 20)", # TF class 13 → Folder "20"
    14: "Gesture_14 (Folder 21)", # TF class 14 → Folder "21"
    15: "Gesture_15 (Folder 22)", # TF class 15 → Folder "22"
    16: "Gesture_16 (Folder 23)", # TF class 16 → Folder "23"
    17: "Gesture_17 (Folder 24)", # TF class 17 → Folder "24"
    18: "Gesture_18 (Folder 25)", # TF class 18 → Folder "25"
    19: "Gesture_19 (Folder 26)", # TF class 19 → Folder "26"
    20: "Gesture_20 (Folder 27)", # TF class 20 → Folder "27"
    21: "Gesture_21 (Folder 28)", # TF class 21 → Folder "28"
    22: "Gesture_22 (Folder 29)", # TF class 22 → Folder "29"
    23: "Gesture_23 (Folder 3)",  # TF class 23 → Folder "3"
    24: "Gesture_24 (Folder 30)", # TF class 24 → Folder "30"
    25: "Gesture_25 (Folder 31)", # TF class 25 → Folder "31"
    26: "Gesture_26 (Folder 32)", # TF class 26 → Folder "32"
    27: "Gesture_27 (Folder 33)", # TF class 27 → Folder "33"
    28: "Gesture_28 (Folder 34)", # TF class 28 → Folder "34"
    29: "Gesture_29 (Folder 35)", # TF class 29 → Folder "35"
    30: "Gesture_30 (Folder 4)",  # TF class 30 → Folder "4"
    31: "Gesture_31 (Folder 5)",  # TF class 31 → Folder "5"
    32: "Gesture_32 (Folder 6)",  # TF class 32 → Folder "6"
    33: "Gesture_33 (Folder 7)",  # TF class 33 → Folder "7"
    34: "Gesture_34 (Folder 8)",  # TF class 34 → Folder "8"
    35: "Gesture_35 (Folder 9)"   # TF class 35 → Folder "9"
}

def create_hybrid_cnn_lstm_model():
    """Recreate the hybrid CNN-LSTM model architecture"""
    print("🏗️  Creating LSTM model architecture...")
    
    # Input for image with exact name
    image_input = tf.keras.Input(shape=(*LSTM_CONFIG['img_size'], 3), name='input_1')
    
    # CNN Feature extraction with exact name
    base_cnn = MobileNetV2(weights='imagenet', include_top=False, 
                         input_shape=(*LSTM_CONFIG['img_size'], 3),
                         name='mobilenetv2_1.00_128')
    base_cnn.trainable = False
    
    # Extract features from image
    cnn_features = base_cnn(image_input)
    
    # Reshape CNN output for LSTM processing
    shape = cnn_features.shape
    reshaped = Reshape((shape[1] * shape[2], shape[3]), name='reshape')(cnn_features)
    
    # Lambda layer - THIS WAS THE MISSING PIECE!
    sequence_features = tf.keras.layers.Lambda(
        lambda x: x[:, :LSTM_CONFIG['sequence_length'], :],
        name='lambda'
    )(reshaped)
    
    # LSTM processing
    lstm_out = LSTM(LSTM_CONFIG['lstm_units'], return_sequences=True, name='lstm')(sequence_features)
    lstm_out = LSTM(LSTM_CONFIG['lstm_units'] // 2, return_sequences=False, name='lstm_1')(lstm_out)
    
    # Also get global features
    global_features = GlobalAveragePooling2D(name='global_average_pooling2d')(cnn_features)
    
    # Combine LSTM output with global features
    combined = tf.keras.layers.concatenate([lstm_out, global_features], name='concatenate')
    
    # Final classification layers
    x = Dense(LSTM_CONFIG['dense_units'], activation='relu', name='dense')(combined)
    x = BatchNormalization(name='batch_normalization')(x)
    x = Dropout(LSTM_CONFIG['dropout_rate'], name='dropout')(x)
    
    x = Dense(LSTM_CONFIG['dense_units'] // 2, activation='relu', name='dense_1')(x)
    x = BatchNormalization(name='batch_normalization_1')(x)
    x = Dropout(LSTM_CONFIG['dropout_rate'], name='dropout_1')(x)
    
    output = Dense(LSTM_CONFIG['num_classes'], activation='softmax', name='dense_2')(x)
    
    model = Model(inputs=image_input, outputs=output)
    return model

def load_lstm_model():
    """Load LSTM model with fallback to architecture recreation"""
    model_paths = [
        'models/lstm_nsl_model.h5',
        'models/lstm_nsl_checkpoint.h5'
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                # Try loading the full model first
                print(f"🔄 Attempting to load full model from {model_path}...")
                model = tf.keras.models.load_model(model_path, compile=False)
                print("✅ Full model loaded successfully!")
                return model, model_path
            except Exception as e:
                print(f"⚠️  Full model loading failed: {e}")
                print("🔄 Recreating model architecture and loading weights...")
                
                try:
                    # Recreate model and load weights
                    model = create_hybrid_cnn_lstm_model()
                    model.load_weights(model_path)
                    print("✅ Model recreated and weights loaded successfully!")
                    return model, model_path
                except Exception as e2:
                    print(f"❌ Failed to recreate model: {e2}")
                    continue
    
    print("❌ No LSTM model found!")
    return None, None

def get_single_sample_from_each_class():
    """Get one random sample from each class using the SAME method as comprehensive test"""
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    # Use the exact same data generator setup as comprehensive test
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        os.path.join(LSTM_CONFIG['dataset_path'], 'test'),
        target_size=LSTM_CONFIG['img_size'],
        batch_size=32,  # Use fixed batch size
        class_mode='categorical',
        shuffle=False
    )
    
    print("📂 Using TensorFlow's class mapping (same as comprehensive test)...")
    print("Class indices:", test_generator.class_indices)
    
    # Get all file paths and labels
    all_files = test_generator.filepaths
    all_labels = test_generator.labels
    
    samples = []
    
    # For each model class index, pick one random sample
    for class_idx in range(LSTM_CONFIG['num_classes']):
        # Find all files for this class
        class_files = [f for i, f in enumerate(all_files) if all_labels[i] == class_idx]
        
        if class_files:
            # Pick a random file
            selected_file = random.choice(class_files)
            
            # Determine which folder this came from
            folder_name = selected_file.split('/')[-2]  # Extract folder name from path
            
            samples.append({
                'class_idx': class_idx,
                'class_name': CLASS_NAMES[class_idx],
                'image_path': selected_file,
                'image_name': selected_file.split('/')[-1],
                'source_folder': folder_name
            })
            print(f"  Model Class {class_idx:2d} ({CLASS_NAMES[class_idx]:<25}): {selected_file.split('/')[-1]} [from folder {folder_name}]")
        else:
            print(f"  ⚠️  No images found for model class {class_idx}")
    
    return samples

def preprocess_image(image_path):
    """Preprocess image for model prediction"""
    try:
        # Load and convert image
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size
        image_resized = image.resize(LSTM_CONFIG['img_size'])
        
        # Convert to array and normalize
        img_array = np.array(image_resized).astype(np.float32) / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_array, axis=0)
        
        return img_batch, image_resized
    except Exception as e:
        print(f"❌ Error preprocessing image {image_path}: {e}")
        return None, None

def test_single_samples(model, samples):
    """Test the model on single samples from each class"""
    print(f"\n🧪 Testing {len(samples)} samples...")
    print("=" * 80)
    
    correct_predictions = 0
    results = []
    
    for i, sample in enumerate(samples):
        print(f"\n🔍 Testing Class {sample['class_idx']:2d}: {sample['class_name']}")
        print(f"   Image: {sample['image_name']}")
        
        # Preprocess image
        processed_image, display_image = preprocess_image(sample['image_path'])
        
        if processed_image is None:
            print("   ❌ Failed to preprocess image")
            continue
        
        # Make prediction
        try:
            predictions = model.predict(processed_image, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            predicted_name = CLASS_NAMES[predicted_class]
            
            # Check if prediction is correct
            is_correct = predicted_class == sample['class_idx']
            if is_correct:
                correct_predictions += 1
                status = "✅ CORRECT"
            else:
                status = "❌ WRONG"
            
            print(f"   Predicted: Class {predicted_class:2d} ({predicted_name}) - {confidence:.3f}")
            print(f"   Result: {status}")
            
            # Store result
            results.append({
                'true_class': sample['class_idx'],
                'true_name': sample['class_name'],
                'predicted_class': predicted_class,
                'predicted_name': predicted_name,
                'confidence': confidence,
                'is_correct': is_correct,
                'image_path': sample['image_path']
            })
            
        except Exception as e:
            print(f"   ❌ Prediction failed: {e}")
    
    return results, correct_predictions

def display_results_summary(results, correct_predictions):
    """Display summary of test results"""
    total_tests = len(results)
    accuracy = (correct_predictions / total_tests) * 100 if total_tests > 0 else 0
    
    print("\n" + "=" * 80)
    print("📊 SINGLE SAMPLE TEST RESULTS SUMMARY")
    print("=" * 80)
    print(f"🎯 Total Samples Tested: {total_tests}")
    print(f"✅ Correct Predictions: {correct_predictions}")
    print(f"❌ Wrong Predictions: {total_tests - correct_predictions}")
    print(f"🏆 Accuracy: {accuracy:.1f}%")
    
    if total_tests > 0:
        # Show wrong predictions if any
        wrong_predictions = [r for r in results if not r['is_correct']]
        
        if wrong_predictions:
            print(f"\n❌ WRONG PREDICTIONS ({len(wrong_predictions)}):")
            print("-" * 60)
            for result in wrong_predictions:
                print(f"   True: {result['true_name']:<15} → Predicted: {result['predicted_name']:<15} ({result['confidence']:.3f})")
        else:
            print("\n🎉 PERFECT SCORE! All predictions were correct!")
        
        # Show confidence statistics
        confidences = [r['confidence'] for r in results]
        avg_confidence = np.mean(confidences)
        min_confidence = np.min(confidences)
        max_confidence = np.max(confidences)
        
        print(f"\n📈 CONFIDENCE STATISTICS:")
        print(f"   Average: {avg_confidence:.3f}")
        print(f"   Minimum: {min_confidence:.3f}")
        print(f"   Maximum: {max_confidence:.3f}")

def main():
    """Main function to run single sample testing"""
    print("🇳🇵 Nepali Sign Language Recognition - Single Sample Test")
    print("=" * 80)
    
    # Load model
    model, model_path = load_lstm_model()
    if model is None:
        print("❌ Cannot proceed without a model!")
        return
    
    print(f"✅ Model loaded from: {model_path}")
    
    # Get single samples from each class
    samples = get_single_sample_from_each_class()
    
    if not samples:
        print("❌ No samples found to test!")
        return
    
    print(f"✅ Found {len(samples)} samples to test")
    
    # Test the samples
    results, correct_predictions = test_single_samples(model, samples)
    
    # Display summary
    display_results_summary(results, correct_predictions)
    
    print("\n🎯 Single Sample Testing Complete!")

if __name__ == "__main__":
    main()

"""
LSTM-based approach for NSL Sign Language Recognition
Treats spatial features as sequential data for better temporal understanding
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Reshape, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt

# LSTM Configuration
LSTM_CONFIG = {
    'dataset_path': 'nslDataset',
    'img_size': (128, 128),  # Smaller for faster processing
    'batch_size': 32,
    'num_classes': 36,
    'epochs': 30,
    'learning_rate': 0.001,
    'sequence_length': 8,  # Number of feature sequences to extract
    'lstm_units': 128,
    'dense_units': 256,
    'dropout_rate': 0.3,
    'model_save_path': 'models/lstm_nsl_model.keras'
}

def setup_gpu():
    """Setup GPU configuration"""
    print("🖥️  Setting up GPU for LSTM training...")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU setup complete - {len(gpus)} GPU(s) found")
        except RuntimeError as e:
            print(f"⚠️  GPU setup warning: {e}")
    else:
        print("💻 No GPU found - using CPU")

def create_data_generators():
    """Create data generators for LSTM training"""
    print("📁 Creating LSTM data generators...")
    
    # Data augmentation optimized for hand gestures
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,  # Moderate rotation for hand gestures
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.1,
        horizontal_flip=False,  # No horizontal flip for sign language
        fill_mode='nearest'
    )
    
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        os.path.join(LSTM_CONFIG['dataset_path'], 'train'),
        target_size=LSTM_CONFIG['img_size'],
        batch_size=LSTM_CONFIG['batch_size'],
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_test_datagen.flow_from_directory(
        os.path.join(LSTM_CONFIG['dataset_path'], 'val'),
        target_size=LSTM_CONFIG['img_size'],
        batch_size=LSTM_CONFIG['batch_size'],
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        os.path.join(LSTM_CONFIG['dataset_path'], 'test'),
        target_size=LSTM_CONFIG['img_size'],
        batch_size=LSTM_CONFIG['batch_size'],
        class_mode='categorical',
        shuffle=False
    )
    
    print(f"✅ LSTM data generators created:")
    print(f"   Training samples: {train_generator.samples}")
    print(f"   Validation samples: {val_generator.samples}")
    print(f"   Test samples: {test_generator.samples}")
    
    return train_generator, val_generator, test_generator

def create_cnn_feature_extractor():
    """Create CNN feature extractor for LSTM input"""
    print("🔧 Creating CNN feature extractor...")
    
    # Use MobileNetV2 as feature extractor (lightweight and efficient)
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(*LSTM_CONFIG['img_size'], 3)
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    # Add custom feature extraction layers
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(256, activation='relu'),
        BatchNormalization()
    ])
    
    print(f"   Feature extractor parameters: {model.count_params():,}")
    return model

def create_lstm_model():
    """Create the main LSTM model for sign language recognition"""
    print("🏗️  Creating LSTM model for sign language recognition...")
    
    model = Sequential([
        # Input layer - expecting sequences of features
        tf.keras.Input(shape=(LSTM_CONFIG['sequence_length'], 256)),
        
        # LSTM layers for temporal understanding
        LSTM(LSTM_CONFIG['lstm_units'], return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
        BatchNormalization(),
        
        LSTM(LSTM_CONFIG['lstm_units'] // 2, return_sequences=False, dropout=0.2, recurrent_dropout=0.2),
        BatchNormalization(),
        
        # Dense layers for classification
        Dense(LSTM_CONFIG['dense_units'], activation='relu'),
        BatchNormalization(),
        Dropout(LSTM_CONFIG['dropout_rate']),
        
        Dense(LSTM_CONFIG['dense_units'] // 2, activation='relu'),
        BatchNormalization(),
        Dropout(LSTM_CONFIG['dropout_rate']),
        
        # Output layer
        Dense(LSTM_CONFIG['num_classes'], activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=LSTM_CONFIG['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"✅ LSTM model created:")
    print(f"   Total parameters: {model.count_params():,}")
    
    return model

def create_spatial_sequence_generator(generator, feature_extractor):
    """
    Convert images into spatial sequences for LSTM
    This treats different regions of the image as sequence steps
    """
    while True:
        batch_images, batch_labels = next(generator)
        batch_size = batch_images.shape[0]
        
        # Extract features from the entire image
        base_features = feature_extractor.predict(batch_images, verbose=0)
        
        # Create sequences by applying slight spatial variations
        sequences = []
        for img in batch_images:
            img_sequences = []
            
            # Create sequence by cropping different regions and extracting features
            for i in range(LSTM_CONFIG['sequence_length']):
                # Apply slight transformations to create sequence variation
                angle = (i - LSTM_CONFIG['sequence_length']//2) * 5  # -15 to +15 degrees
                zoom = 1.0 + (i * 0.02)  # Slight zoom changes
                
                # Apply transformation
                transformed_img = tf.image.rot90(tf.expand_dims(img, 0), k=0)  # Keep original for now
                transformed_img = tf.image.resize(transformed_img, LSTM_CONFIG['img_size'])
                
                # Extract features
                features = feature_extractor.predict(transformed_img, verbose=0)
                img_sequences.append(features[0])
            
            sequences.append(img_sequences)
        
        yield np.array(sequences), batch_labels

def create_hybrid_cnn_lstm_model():
    """
    Create a hybrid CNN-LSTM model that processes images as spatial sequences
    """
    print("🚀 Creating Hybrid CNN-LSTM model...")
    
    # Input for image
    image_input = tf.keras.Input(shape=(*LSTM_CONFIG['img_size'], 3))
    
    # CNN Feature extraction
    base_cnn = MobileNetV2(weights='imagenet', include_top=False, input_shape=(*LSTM_CONFIG['img_size'], 3))
    base_cnn.trainable = False
    
    # Extract features from image
    cnn_features = base_cnn(image_input)
    
    # Reshape CNN output for LSTM processing
    # Treat spatial dimensions as sequence
    shape = cnn_features.shape
    reshaped = Reshape((shape[1] * shape[2], shape[3]))(cnn_features)  # (height*width, channels)
    
    # Take only a subset for sequence processing (to manage memory)
    sequence_features = tf.keras.layers.Lambda(lambda x: x[:, :LSTM_CONFIG['sequence_length'], :])(reshaped)
    
    # LSTM processing
    lstm_out = LSTM(LSTM_CONFIG['lstm_units'], return_sequences=True)(sequence_features)
    lstm_out = LSTM(LSTM_CONFIG['lstm_units'] // 2, return_sequences=False)(lstm_out)
    
    # Also get global features
    global_features = GlobalAveragePooling2D()(cnn_features)
    
    # Combine LSTM output with global features
    combined = tf.keras.layers.concatenate([lstm_out, global_features])
    
    # Final classification layers
    x = Dense(LSTM_CONFIG['dense_units'], activation='relu')(combined)
    x = BatchNormalization()(x)
    x = Dropout(LSTM_CONFIG['dropout_rate'])(x)
    
    x = Dense(LSTM_CONFIG['dense_units'] // 2, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(LSTM_CONFIG['dropout_rate'])(x)
    
    output = Dense(LSTM_CONFIG['num_classes'], activation='softmax')(x)
    
    model = Model(inputs=image_input, outputs=output)
    
    model.compile(
        optimizer=Adam(learning_rate=LSTM_CONFIG['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"✅ Hybrid CNN-LSTM model created:")
    print(f"   Total parameters: {model.count_params():,}")
    print(f"   Trainable parameters: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")
    
    return model

def get_callbacks():
    """Create callbacks for training"""
    os.makedirs('models', exist_ok=True)
    
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            'models/lstm_nsl_checkpoint.h5',  # Use H5 format for compatibility
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
            save_format='h5'  # Explicitly specify H5 format
        )
    ]
    
    return callbacks

def train_lstm_model():
    """Main training function for LSTM model"""
    print("🎯 LSTM-based NSL Sign Language Recognition")
    print("=" * 60)
    
    # Setup
    setup_gpu()
    
    # Create data generators
    train_gen, val_gen, test_gen = create_data_generators()
    
    # Create hybrid model (easier to work with)
    model = create_hybrid_cnn_lstm_model()
    
    # Print model summary
    print("\n📋 Model Architecture:")
    model.summary()
    
    # Create callbacks
    callbacks = get_callbacks()
    
    print(f"\n🔥 Starting LSTM training...")
    print(f"   Epochs: {LSTM_CONFIG['epochs']}")
    print(f"   Batch size: {LSTM_CONFIG['batch_size']}")
    print(f"   Learning rate: {LSTM_CONFIG['learning_rate']}")
    print(f"   LSTM units: {LSTM_CONFIG['lstm_units']}")
    
    # Train the model
    history = model.fit(
        train_gen,
        epochs=LSTM_CONFIG['epochs'],
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"\n📊 Evaluating on test set...")
    test_results = model.evaluate(test_gen, verbose=1)
    
    print(f"\n🎯 Final Results:")
    print(f"   Test Loss: {test_results[0]:.4f}")
    print(f"   Test Accuracy: {test_results[1]:.4f} ({test_results[1]*100:.2f}%)")
    
    # Save final model
    print(f"\n💾 Saving final model...")
    try:
        # Try H5 format first (more compatible)
        h5_path = LSTM_CONFIG['model_save_path'].replace('.keras', '.h5')
        model.save(h5_path, save_format='h5')
        print(f"✅ Model saved as H5: {h5_path}")
    except Exception as e:
        print(f"⚠️  H5 format failed: {e}")
        try:
            # Fallback to Keras format without problematic options
            model.save_weights('models/lstm_nsl_weights.h5')
            print(f"✅ Model weights saved: models/lstm_nsl_weights.h5")
            print("ℹ️  You can reload using model.load_weights('models/lstm_nsl_weights.h5')")
        except Exception as e2:
            print(f"❌ All save methods failed: {e2}")
    
    # Plot training history
    plot_lstm_training_history(history)
    
    return model, history

def plot_lstm_training_history(history):
    """Plot training history for LSTM model"""
    print("📈 Creating training plots...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
    ax1.set_title('LSTM Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss', color='blue')
    ax2.plot(history.history['val_loss'], label='Validation Loss', color='red')
    ax2.set_title('LSTM Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('models/lstm_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📈 Training plots saved as 'models/lstm_training_history.png'")

def test_lstm_predictions(model, test_gen, num_samples=5):
    """Test LSTM model predictions on random samples"""
    print(f"\n🧪 Testing LSTM predictions on {num_samples} samples...")
    
    # Get some test data
    test_images, test_labels = next(test_gen)
    
    # Make predictions
    predictions = model.predict(test_images[:num_samples])
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(test_labels[:num_samples], axis=1)
    
    # Display results
    plt.figure(figsize=(15, 3))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(test_images[i])
        
        is_correct = predicted_classes[i] == true_classes[i]
        color = 'green' if is_correct else 'red'
        status = '✅' if is_correct else '❌'
        confidence = predictions[i][predicted_classes[i]]
        
        plt.title(f'{status} True: {true_classes[i]}\nPred: {predicted_classes[i]} ({confidence:.2f})', 
                 color=color, fontsize=10)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    accuracy = np.mean(predicted_classes == true_classes)
    print(f"📊 Sample accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    # Check dataset
    if not os.path.exists(LSTM_CONFIG['dataset_path']):
        print(f"❌ Dataset not found at {LSTM_CONFIG['dataset_path']}")
        exit(1)
    
    try:
        print("🚀 Starting LSTM-based NSL training...")
        model, history = train_lstm_model()
        
        print("\n🧪 Testing model predictions...")
        train_gen, val_gen, test_gen = create_data_generators()
        test_lstm_predictions(model, test_gen)
        
        print("\n✅ LSTM training completed successfully!")
        print(f"🎯 Model saved at: {LSTM_CONFIG['model_save_path']}")
        
    except Exception as e:
        print(f"\n❌ LSTM training failed: {e}")
        import traceback
        traceback.print_exc()

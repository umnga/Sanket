"""
Comprehensive testing script for LSTM NSL model
Handles the Lambda layer loading issue by recreating the model architecture
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Reshape, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# LSTM Configuration (same as training)
LSTM_CONFIG = {
    'dataset_path': 'nslDataset',
    'img_size': (128, 128),
    'batch_size': 32,
    'num_classes': 36,
    'sequence_length': 8,
    'lstm_units': 128,
    'dense_units': 256,
    'dropout_rate': 0.3,
}

def create_hybrid_cnn_lstm_model():
    """
    Recreate the hybrid CNN-LSTM model architecture with exact layer names
    """
    print("🏗️  Recreating LSTM model architecture...")
    
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
    
    # LSTM processing with exact names
    lstm_out = LSTM(LSTM_CONFIG['lstm_units'], return_sequences=True, name='lstm')(sequence_features)
    lstm_out = LSTM(LSTM_CONFIG['lstm_units'] // 2, return_sequences=False, name='lstm_1')(lstm_out)
    
    # Also get global features with exact name
    global_features = GlobalAveragePooling2D(name='global_average_pooling2d')(cnn_features)
    
    # Combine LSTM output with global features
    combined = tf.keras.layers.concatenate([lstm_out, global_features], name='concatenate')
    
    # Final classification layers with exact names
    x = Dense(LSTM_CONFIG['dense_units'], activation='relu', name='dense')(combined)
    x = BatchNormalization(name='batch_normalization')(x)
    x = Dropout(LSTM_CONFIG['dropout_rate'], name='dropout')(x)
    
    x = Dense(LSTM_CONFIG['dense_units'] // 2, activation='relu', name='dense_1')(x)
    x = BatchNormalization(name='batch_normalization_1')(x)
    x = Dropout(LSTM_CONFIG['dropout_rate'], name='dropout_1')(x)
    
    output = Dense(LSTM_CONFIG['num_classes'], activation='softmax', name='dense_2')(x)
    
    model = Model(inputs=image_input, outputs=output)
    return model

def load_lstm_model_with_weights(weights_path):
    """
    Load LSTM model by recreating architecture and loading weights
    """
    try:
        # Try loading the full model first (might work)
        print(f"🔄 Attempting to load full model from {weights_path}...")
        model = tf.keras.models.load_model(weights_path, compile=False)
        print("✅ Full model loaded successfully!")
        return model
    except Exception as e:
        print(f"⚠️  Full model loading failed: {e}")
        print("🔄 Recreating model architecture and loading weights...")
        
        try:
            # Recreate model and load weights
            model = create_hybrid_cnn_lstm_model()
            
            # Try to load weights
            if weights_path.endswith('.h5'):
                model.load_weights(weights_path)
            else:
                # For keras format, we'll try loading checkpoint instead
                checkpoint_path = 'models/lstm_nsl_checkpoint.h5'
                if os.path.exists(checkpoint_path):
                    model.load_weights(checkpoint_path)
                else:
                    raise Exception("No compatible weights file found")
            
            print("✅ Model recreated and weights loaded successfully!")
            return model
            
        except Exception as e2:
            print(f"❌ Failed to recreate model: {e2}")
            return None

def create_test_data_generator():
    """Create test data generator"""
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        os.path.join(LSTM_CONFIG['dataset_path'], 'test'),
        target_size=LSTM_CONFIG['img_size'],
        batch_size=LSTM_CONFIG['batch_size'],
        class_mode='categorical',
        shuffle=False
    )
    
    return test_generator

def comprehensive_lstm_evaluation(model, test_generator):
    """
    Comprehensive evaluation of LSTM model
    """
    print("\n" + "="*70)
    print("📊 COMPREHENSIVE LSTM MODEL EVALUATION")
    print("="*70)
    
    # Get all test data
    print("🔄 Collecting test data...")
    
    # Reset generator
    test_generator.reset()
    
    # Get predictions for all test data
    print("🧠 Making predictions on all test data...")
    predictions = model.predict(test_generator, verbose=1)
    
    # Get true labels
    true_labels = test_generator.classes
    predicted_labels = np.argmax(predictions, axis=1)
    
    print(f"📈 Total test samples: {len(true_labels)}")
    print(f"📈 Number of classes: {len(np.unique(true_labels))}")
    
    # Overall accuracy
    accuracy = np.mean(predicted_labels == true_labels)
    print(f"\n🎯 Overall Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Detailed classification report
    print("\n📊 DETAILED CLASSIFICATION REPORT:")
    print("="*80)
    
    class_names = [f"Class_{i}" for i in range(LSTM_CONFIG['num_classes'])]
    report = classification_report(
        true_labels, 
        predicted_labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    # Print the report
    print(classification_report(
        true_labels, 
        predicted_labels,
        target_names=class_names,
        zero_division=0
    ))
    
    # Analyze per-class performance
    print("\n🎯 PER-CLASS PERFORMANCE ANALYSIS:")
    print("="*80)
    
    class_metrics = []
    for i in range(LSTM_CONFIG['num_classes']):
        class_key = f"Class_{i}"
        if class_key in report:
            metrics = report[class_key]
            class_metrics.append({
                'class': i,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1-score'],
                'support': metrics['support']
            })
    
    # Sort by F1-score
    class_metrics_sorted = sorted(class_metrics, key=lambda x: x['f1_score'])
    
    print("🔥 TOP 5 BEST PERFORMING CLASSES:")
    for metrics in class_metrics_sorted[-5:]:
        print(f"   Class {metrics['class']:2d}: F1={metrics['f1_score']:.3f}, "
              f"Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, "
              f"Support={int(metrics['support'])}")
    
    print("\n⚠️  TOP 5 WORST PERFORMING CLASSES:")
    for metrics in class_metrics_sorted[:5]:
        print(f"   Class {metrics['class']:2d}: F1={metrics['f1_score']:.3f}, "
              f"Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, "
              f"Support={int(metrics['support'])}")
    
    # Confusion Matrix
    print("\n🎨 Generating confusion matrix...")
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # Plot confusion matrix
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=False, cmap='Blues', 
               xticklabels=[f'C{i}' for i in range(LSTM_CONFIG['num_classes'])],
               yticklabels=[f'C{i}' for i in range(LSTM_CONFIG['num_classes'])])
    plt.title('LSTM Model - Confusion Matrix for NSL Sign Language Classification')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.tight_layout()
    plt.savefig('models/lstm_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Per-class accuracy from confusion matrix
    print(f"\n🎯 CLASS-WISE ACCURACY FROM CONFUSION MATRIX:")
    class_accuracies = []
    for i in range(LSTM_CONFIG['num_classes']):
        if i < cm.shape[0]:
            true_positives = cm[i, i] if i < cm.shape[1] else 0
            total_true = cm[i, :].sum() if cm[i, :].sum() > 0 else 1
            class_accuracy = true_positives / total_true
            class_accuracies.append((i, class_accuracy, int(total_true)))
    
    # Sort by accuracy
    class_accuracies.sort(key=lambda x: x[1])
    
    print("   📉 Worst 10 classes:")
    for class_num, acc, support in class_accuracies[:10]:
        print(f"     Class {class_num:2d}: {acc:.3f} ({acc*100:.1f}%) - {support} samples")
    
    print("   📈 Best 10 classes:")
    for class_num, acc, support in class_accuracies[-10:]:
        print(f"     Class {class_num:2d}: {acc:.3f} ({acc*100:.1f}%) - {support} samples")
    
    # Summary statistics
    precisions = [m['precision'] for m in class_metrics if m['precision'] > 0]
    recalls = [m['recall'] for m in class_metrics if m['recall'] > 0]
    f1_scores = [m['f1_score'] for m in class_metrics if m['f1_score'] > 0]
    
    print(f"\n📊 SUMMARY STATISTICS:")
    print(f"   Average Precision: {np.mean(precisions):.3f} ± {np.std(precisions):.3f}")
    print(f"   Average Recall: {np.mean(recalls):.3f} ± {np.std(recalls):.3f}")
    print(f"   Average F1-Score: {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")
    print(f"   Macro Avg F1: {report['macro avg']['f1-score']:.3f}")
    print(f"   Weighted Avg F1: {report['weighted avg']['f1-score']:.3f}")
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm,
        'class_metrics': class_metrics
    }

def test_sample_predictions(model, test_generator, num_samples=10):
    """
    Test sample predictions with visualization
    """
    print(f"\n🧪 TESTING {num_samples} SAMPLE PREDICTIONS")
    print("="*50)
    
    # Get some test samples
    test_generator.reset()
    batch_images, batch_labels = next(test_generator)
    
    # Select random samples
    indices = np.random.choice(len(batch_images), size=min(num_samples, len(batch_images)), replace=False)
    
    sample_images = batch_images[indices]
    sample_labels = batch_labels[indices]
    
    # Make predictions
    predictions = model.predict(sample_images, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(sample_labels, axis=1)
    
    # Display results
    cols = min(5, num_samples)
    rows = (num_samples + cols - 1) // cols
    
    plt.figure(figsize=(4*cols, 3*rows))
    
    correct_count = 0
    for i in range(num_samples):
        plt.subplot(rows, cols, i+1)
        plt.imshow(sample_images[i])
        
        is_correct = predicted_classes[i] == true_classes[i]
        if is_correct:
            correct_count += 1
            color = 'green'
            status = '✅'
        else:
            color = 'red'
            status = '❌'
        
        confidence = predictions[i][predicted_classes[i]]
        
        # Get top 3 predictions
        top3_indices = np.argsort(predictions[i])[-3:][::-1]
        top3_text = f"Top3: {top3_indices[0]},{top3_indices[1]},{top3_indices[2]}"
        
        plt.title(f"{status} True: {true_classes[i]} | Pred: {predicted_classes[i]}\n"
                 f"Conf: {confidence:.2f} | {top3_text}", 
                 color=color, fontsize=9)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('models/lstm_sample_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    sample_accuracy = correct_count / num_samples
    print(f"📊 Sample Test Results: {correct_count}/{num_samples} correct ({sample_accuracy:.1%})")

def main():
    """Main function to run comprehensive LSTM evaluation"""
    print("🎯 LSTM Model Comprehensive Testing")
    print("="*60)
    
    # Check if dataset exists
    if not os.path.exists(LSTM_CONFIG['dataset_path']):
        print(f"❌ Dataset not found at {LSTM_CONFIG['dataset_path']}")
        return
    
    # Try to load the model
    model_paths = [
        'models/lstm_nsl_model.h5',
        'models/lstm_nsl_checkpoint.h5'
    ]
    
    model = None
    for path in model_paths:
        if os.path.exists(path):
            print(f"🔍 Found model: {path}")
            model = load_lstm_model_with_weights(path)
            if model is not None:
                break
    
    if model is None:
        print("❌ Could not load any LSTM model")
        return
    
    # Create test data generator
    print("\n📁 Creating test data generator...")
    test_generator = create_test_data_generator()
    print(f"✅ Test generator created with {test_generator.samples} samples")
    
    # Run sample predictions first
    print("\n" + "="*60)
    test_sample_predictions(model, test_generator, num_samples=12)
    
    # Run comprehensive evaluation
    print("\n" + "="*60)
    results = comprehensive_lstm_evaluation(model, test_generator)
    
    print(f"\n✅ LSTM Model Testing Complete!")
    print(f"🎯 Final Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"📊 Results saved to models/lstm_confusion_matrix.png")

if __name__ == "__main__":
    main()

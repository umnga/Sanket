# NSL Sign Language Recognition - LSTM Model

Real-time NSL (Numeric Sign Language) recognition using LSTM neural networks and modern web interface.

## 🎯 Project Overview

This project uses a hybrid CNN-LSTM architecture to recognize American Sign Language (ASL) numeric signs (0-35) through an intuitive web interface. The model achieved **99.83% accuracy** on the test dataset!

## 📁 Project Structure

```
sanket/
├── models/                          # Trained models and results
│   ├── lstm_nsl_model.h5           # Main LSTM model (99.83% accuracy)
│   ├── lstm_nsl_checkpoint.h5      # Backup checkpoint
│   ├── lstm_confusion_matrix.png   # Model performance visualization
│   ├── lstm_sample_predictions.png # Sample prediction results
│   └── lstm_training_history.png   # Training progress charts
├── nslDataset/                      # Training dataset
│   ├── train/                       # Training images (36 classes)
│   ├── val/                         # Validation images
│   └── test/                        # Test images
├── nsl_clean_env/                   # Python virtual environment
├── streamlit_app.py                 # 🌐 Main web application
├── train_lstm_nsl.py               # Model training script
├── test_lstm_comprehensive.py      # Model evaluation script
└── requirements.txt                 # Dependencies
```

## 🚀 Quick Start

### 1. Activate Environment
```bash
source nsl_clean_env/bin/activate
```

### 2. Run Web Application
```bash
streamlit run streamlit_app.py
```

### 3. Using the Web Interface

**Features:**
- **📤 Upload Images**: Drag & drop or browse for sign language images
- **🎯 Instant Predictions**: Get results with confidence scores
- **📊 Detailed Analysis**: View top predictions and confidence breakdown
- **📈 Model Performance**: See comprehensive accuracy metrics
- **🧪 Sample Results**: View test predictions and model architecture

**Instructions:**
1. Open the web app in your browser (usually `http://localhost:8501`)
2. Navigate to the "Predict" tab
3. Upload a clear image of a sign language gesture
4. View the prediction with confidence score
5. Explore detailed analysis and alternative predictions

## 📊 Model Performance

- **Overall Accuracy**: 99.83%
- **Perfect Classes**: 27 out of 36 classes achieved 100% accuracy
- **Lowest Accuracy**: 97.8% (Class 1)
- **Average F1-Score**: 99.8% ± 0.3%

### Top Performing Classes
- Classes 27, 28, 29, 32, 35: Perfect 100% accuracy
- Most classes: 99.6% - 100% accuracy

### Model Architecture
- **Base**: MobileNetV2 (pre-trained on ImageNet)
- **Sequence Processing**: Dual LSTM layers (128 → 64 units)
- **Feature Fusion**: Combined spatial and sequential features
- **Image Size**: 128×128 pixels for efficient processing
- **Classes**: 36 (representing signs 0-35)

## 🛠️ Technical Details

### Dependencies
- TensorFlow 2.15+ with Metal acceleration (Apple Silicon)
- Streamlit 1.28+ for web interface
- NumPy, Matplotlib, Scikit-learn, Pandas
- Python 3.11

### Key Features
- **Modern Web Interface**: Intuitive Streamlit-based UI
- **Upload & Predict**: Simple drag-and-drop image upload
- **Real-time Analysis**: Instant predictions with confidence scores
- **Visual Feedback**: Detailed charts and performance metrics
- **Model Performance**: Comprehensive accuracy breakdowns
- **Multi-tab Interface**: Organized sections for different functions

## 📈 Training & Evaluation

### Training the Model
```bash
python train_lstm_nsl.py
```

### Comprehensive Evaluation
```bash
python test_lstm_comprehensive.py
```

This will generate:
- Detailed classification report
- Confusion matrix visualization
- Per-class performance analysis
- Sample prediction visualizations

## � Streamlit Web Interface Features

### Multi-Tab Interface
- **🎯 Predict Tab**: Upload images and get instant predictions
- **📊 Performance Tab**: View model metrics and visualizations  
- **🧪 Sample Results Tab**: Explore test predictions and architecture

### Prediction Features
- **Image Upload**: Drag & drop or browse for images
- **Instant Results**: Prediction with confidence percentage
- **Top Predictions**: View alternative interpretations
- **Confidence Analysis**: Color-coded confidence indicators
- **Detailed Charts**: Bar charts and tables of all predictions

### Performance Insights
- **Real-time Metrics**: Live accuracy and F1-score displays
- **Confusion Matrix**: Visual representation of model performance
- **Training History**: Charts showing model learning progress
- **Class Breakdown**: Per-class accuracy analysis

## 🔧 Troubleshooting

### Common Issues

1. **Streamlit not starting**
   - Ensure environment is activated
   - Check if port 8501 is available
   - Try: `streamlit run streamlit_app.py --server.port 8502`
   
2. **Model loading errors**
   - Verify model files exist in `models/` directory
   - Check TensorFlow installation
   
3. **Poor predictions**
   - Ensure good lighting in uploaded images
   - Use clear, high-resolution images
   - Position hand clearly against plain background

### Performance Optimization
- Model uses GPU acceleration on Apple Silicon
- Image preprocessing is optimized for 128×128 input
- Caching ensures fast repeated predictions

## 📚 Dataset Information

- **Source**: NSL (Numeric Sign Language) dataset
- **Classes**: 36 sign language gestures (0-35)
- **Training Images**: ~28,800 images
- **Validation Images**: ~7,200 images  
- **Test Images**: 8,100 images (225 per class)

## 🏆 Model Achievements

✅ **99.83% Test Accuracy** - Exceptional performance
✅ **Real-time Processing** - Instant web-based predictions
✅ **Modern UI** - Beautiful Streamlit interface
✅ **Upload & Predict** - Simple drag-and-drop functionality
✅ **Production Ready** - Optimized for deployment

## 📞 Usage Notes

- Upload clear, well-lit images for best accuracy
- Use plain backgrounds when possible
- Practice standard ASL numeric signs for best results
- The model was trained on the specific NSL dataset format

---

**Created with LSTM neural networks and modern Streamlit interface! 🤖🌐**

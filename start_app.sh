#!/bin/bash

# NSL Sign Language Recognition - Quick Launcher
# Streamlit Web Application

echo "🎯 NSL Sign Language Recognition - LSTM Model"
echo "=============================================="
echo ""
echo "🚀 Starting Streamlit web application..."
echo ""

# Check if virtual environment exists
if [ ! -d "nsl_clean_env" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run setup first."
    exit 1
fi

# Check if model exists
if [ ! -f "models/lstm_nsl_model.h5" ] && [ ! -f "models/lstm_nsl_checkpoint.h5" ]; then
    echo "❌ LSTM model not found!"
    echo "Please ensure model files exist in models/ directory."
    exit 1
fi

# Activate environment and start Streamlit
source nsl_clean_env/bin/activate

echo "✅ Environment activated"
echo "🌐 Starting web interface..."
echo ""
echo "📱 Open your browser and go to: http://localhost:8501"
echo ""
echo "🛑 Press Ctrl+C to stop the application"
echo ""

streamlit run streamlit_app.py

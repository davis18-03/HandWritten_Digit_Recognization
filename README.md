# Handwritten Digit Recognition

A Python application that uses deep learning to recognize handwritten digits in real-time. Built with TensorFlow/Keras and featuring an intuitive GUI interface built with Tkinter.

## Features

* Real-time digit recognition with confidence scoring and thresholding
* Interactive drawing canvas with adjustable brush
* Undo/Redo functionality
* Eraser tool
* Save drawings as PNG files
* Advanced preprocessing for improved accuracy
* Robust error handling and comprehensive logging
* GPU acceleration support (when available)
* Model verification on startup

## Requirements

* Python 3.8+
* TensorFlow 2.x
* NumPy
* Pillow (PIL)
* tkinter (usually comes with Python)

## 🚀 Installation

1. Clone the repository:

```bash
git clone https://github.com/davis18-03/digit-recognition.git
cd digit-recognition
```

2. Install the required packages:

```bash
pip install tensorflow numpy pillow
```

## Usage

Run the application using:

```bash
python main.py
```

The application will:

1. Load or train a CNN model on first run (this may take a few minutes)
2. Open a GUI window where you can:
   * Draw digits using your mouse
   * Use the eraser tool to correct mistakes
   * Undo/redo your drawing actions
   * Save your drawings as PNG files
   * Get real-time predictions with confidence scores

## Project Structure

* `main.py` - Entry point of the application
* `gui_app.py` - GUI implementation using Tkinter
* `model_handler.py` - Neural network model management
* `utils.py` - Utility functions (logging setup)
* `tests/` - Unit tests
    * `test_model_handler.py` - Tests for model functionality

## Technical Details

### Neural Network Architecture

The model uses a Convolutional Neural Network (CNN) with:

* Two convolutional blocks with dropout layers for regularization
* Flatten layer for dimension reduction
* Dense layers with dropout for robust classification
* Training on the MNIST dataset with validation
* Model verification on loading
* Automatic retraining if model is corrupted

### Image Processing Pipeline

* Adaptive thresholding with noise reduction
* Smart bounding box detection
* Aspect ratio preservation
* Dynamic padding and centering
* Multi-stage preprocessing for optimal recognition

### Recognition Features

* Confidence thresholding (0.3 minimum)
* Real-time prediction feedback
* GPU acceleration when available
* Comprehensive error handling
* Detailed logging of prediction process

### GUI Features

* 300x300 drawing canvas
* Real-time drawing feedback
* Visual confidence indicators
* Progress indication during prediction
* Tooltips for all controls
* Responsive interface with input validation

## Testing

Run the unit tests using:

```bash
python -m unittest tests/test_model_handler.py
```

## Logging

The application maintains detailed logs in the `logs` directory, including:

* Model training progress and validation metrics
* TensorFlow version and GPU availability
* Prediction results with confidence scores
* Preprocessing stages and image statistics
* Detailed error traces with context
* Model verification and health checks
* Application status and performance metrics

## License
This is a personal project. All rights reserved.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Troubleshooting

Common issues:

* If the model file (`mnist.h5`) is corrupted, delete it and restart the application to retrain
* For GPU acceleration, ensure CUDA and cuDNN are properly installed
* Check the logs in the `logs` directory for detailed error messages

## Future Improvements

* Support for different model architectures
* Data augmentation for better accuracy
* Export trained models
* Support for different drawing tools
* Batch prediction support
* Dark mode support

import os
import logging
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Log GPU availability at startup

logger = logging.getLogger(__name__)

class ModelHandler:
    """Handles loading, training, and predicting with the MNIST model."""

    MODEL_FILE = 'mnist.h5'
    IMAGE_SIZE = (28, 28)  # MNIST standard size
    INPUT_SHAPE = (28, 28, 1)  # Shape expected by the model

    def __init__(self):
        # Log TensorFlow version and GPU availability
        logger.info(f"TensorFlow version: {tf.__version__}")
        gpus = tf.config.list_physical_devices('GPU')
        logger.info(f"GPU Devices: {gpus}")
        
        self.model = None
        self._load_or_train_model()

    def _load_or_train_model(self):
        """Loads the model if it exists, otherwise trains and saves a new one."""
        if os.path.exists(self.MODEL_FILE):
            try:
                logger.info(f"Loading model from '{self.MODEL_FILE}'...")
                self.model = load_model(self.MODEL_FILE, compile=True)
                
                # Verify the model is working by testing it
                test_input = np.zeros((1, *self.INPUT_SHAPE))
                _ = self.model.predict(test_input, verbose=0)
                
                logger.info(f"Model '{self.MODEL_FILE}' loaded and verified successfully.")
                return
            except Exception as e:
                logger.error(f"Failed to load model from '{self.MODEL_FILE}': {e}. Attempting to retrain.", exc_info=True)
                if os.path.exists(self.MODEL_FILE):
                    os.remove(self.MODEL_FILE)
                    logger.info(f"Removed corrupted model file: {self.MODEL_FILE}")
        
        logger.info(f"Training a new MNIST model...")
        self._train_and_save_model()

    def _train_and_save_model(self):
        """Trains a CNN model on MNIST data and saves it."""
        try:
            # Load MNIST data
            (x_train, y_train), (x_test, y_test) = mnist.load_data()

            # Preprocess data
            x_train = x_train.reshape(x_train.shape[0], *self.IMAGE_SIZE, 1).astype('float32') / 255
            x_test = x_test.reshape(x_test.shape[0], *self.IMAGE_SIZE, 1).astype('float32') / 255
            y_train = to_categorical(y_train, num_classes=10)
            y_test = to_categorical(y_test, num_classes=10)

            # Build a simple effective CNN model
            model = Sequential([
                # First convolutional block
                Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(*self.IMAGE_SIZE, 1)),
                Conv2D(64, (3, 3), activation='relu'),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                
                # Second convolutional block
                Conv2D(64, (3, 3), activation='relu'),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                
                # Dense layers
                Flatten(),
                Dense(128, activation='relu'),
                Dropout(0.5),
                Dense(10, activation='softmax')
            ])

            # Compile and train the model
            model.compile(loss='categorical_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy'])
            
            # Train the model
            logger.info("Starting model training (this may take a few minutes)...")
            model.fit(x_train, y_train,
                     batch_size=128,
                     epochs=10,
                     validation_data=(x_test, y_test),
                     verbose=1)
            logger.info("Model training complete.")

            # Evaluate model
            loss, acc = model.evaluate(x_test, y_test, verbose=0)
            logger.info(f"Model evaluation: Loss={loss:.4f}, Accuracy={acc*100:.2f}%")

            # Save the model
            model.save(self.MODEL_FILE)
            self.model = model
            logger.info(f"Model '{self.MODEL_FILE}' trained and saved successfully.")
        except Exception as e:
            logger.critical(f"Failed to train and save model: {e}", exc_info=True)
            raise # Re-raise to stop application if model can't be prepared

    def predict_digit(self, img: 'Image') -> tuple[int, float]:
        """
        Predicts the digit from a pre-processed image.
        Args:
            img: A PIL Image object (expected to be grayscale, but will convert).
        Returns:
            A tuple containing the predicted digit (int) and its confidence (float).
        Raises:
            Exception: If preprocessing or prediction fails.
        """
        if self.model is None:
            logger.error("Prediction requested but model is not loaded or trained.")
            return -1, 0.0  # Indicate an error state

        try:
            # Input validation
            if img is None:
                logger.error("Received None image")
                return -1, 0.0

            logger.debug("Starting image preprocessing...")
            img = img.convert('L')  # Convert to grayscale
            
            # First resize to a manageable size while maintaining aspect ratio
            img.thumbnail((200, 200), Image.Resampling.LANCZOS)
            
            # Convert to numpy array and check for empty image
            arr = np.array(img)
            if arr.size == 0:
                logger.warning("Empty image received")
                return -1, 0.0
            
            # Adaptive thresholding with noise reduction
            mask = arr < 250  # Only consider actual drawn pixels
            if not np.any(mask):
                logger.warning("No drawn pixels detected in image")
                return -1, 0.0
                
            threshold = np.mean(arr[mask])
            arr = np.where(arr < threshold, 0, 255)
            
            # Find bounding box of the digit
            coords = np.column_stack(np.where(arr < 255))
            if not coords.size:
                logger.warning("No digit detected after thresholding")
                return -1, 0.0
            
            # Extract and pad the digit
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0)
            digit = arr[y0:y1+1, x0:x1+1]
            
            # Add padding to make it square
            h, w = digit.shape
            size = max(h, w) + 20  # Add padding
            
            # Create a square image with the digit centered
            new_img = np.ones((size, size), dtype=np.uint8) * 255
            y_start = (size - h) // 2
            x_start = (size - w) // 2
            new_img[y_start:y_start+h, x_start:x_start+w] = digit
            
            # Convert back to PIL Image for proper resizing
            new_img = Image.fromarray(new_img)
            new_img = new_img.resize(self.IMAGE_SIZE, Image.Resampling.LANCZOS)
            arr = np.array(new_img)
            
            # Normalize and reshape for model input
            img_array = arr.reshape(1, *self.INPUT_SHAPE).astype('float32') / 255.0
            
            # Make prediction
            logger.debug("Running model prediction...")
            res = self.model.predict(img_array, verbose=0)[0]
            predicted_digit = int(np.argmax(res))
            confidence = float(max(res))
            
            # Log prediction details
            logger.info(f"Predicted digit: {predicted_digit} with {confidence*100:.2f}% confidence")
            
            # Only return high confidence predictions
            if confidence < 0.3:
                logger.warning(f"Low confidence prediction: {confidence*100:.2f}%")
                return -1, confidence
                
            return predicted_digit, confidence
            
        except Exception as e:
            logger.error(f"Error during prediction process: {e}", exc_info=True)
            return -1, 0.0
        
        # First resize to a manageable size while maintaining aspect ratio
        img.thumbnail((200, 200), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and binarize
        arr = np.array(img)
        if arr.size > 0:  # Check if image is not empty
            # Adaptive thresholding
            threshold = np.mean(arr[arr < 250])  # Only consider actual drawn pixels
            arr = np.where(arr < threshold, 0, 255)
            
            # Find bounding box of the digit
            coords = np.column_stack(np.where(arr < 255))
            if coords.size:
                y0, x0 = coords.min(axis=0)
                y1, x1 = coords.max(axis=0)
                digit = arr[y0:y1+1, x0:x1+1]
                
                # Add padding to make it square
                h, w = digit.shape
                size = max(h, w) + 20  # Add padding
                
                # Create a square image with the digit centered
                new_img = np.ones((size, size), dtype=np.uint8) * 255
                y_start = (size - h) // 2
                x_start = (size - w) // 2
                new_img[y_start:y_start+h, x_start:x_start+w] = digit
                
                # Convert back to PIL Image for proper resizing
                new_img = Image.fromarray(new_img)
                new_img = new_img.resize(self.IMAGE_SIZE, Image.Resampling.LANCZOS)
                arr = np.array(new_img)
            
        # Ensure proper shape for the model
        img_array = arr.reshape(1, *self.INPUT_SHAPE).astype('float32') / 255.0
        
        try:
            res = self.model.predict(img_array, verbose=0)[0]
            predicted_digit = int(np.argmax(res))
            confidence = float(max(res))
            logger.debug(f"Predicted: {predicted_digit}, Confidence: {confidence*100:.2f}%")
            return predicted_digit, confidence
        except Exception as e:
            logger.error(f"Error during prediction: {e}", exc_info=True)
            return -1, 0.0  # Indicate an error state
        # Create default array for empty images
        if not coords.size:
            logger.warning("No digit detected in the image")
            return -1, 0.0

        # Return early if preprocessing failed
        if arr is None:
            logger.error("Image preprocessing failed")
            return -1, 0.0
import unittest
import os
import numpy as np
from PIL import Image
from model_handler import ModelHandler

# Suppress TensorFlow warnings/logs during tests
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # For TensorFlow 2.x

class TestModelHandler(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up for all tests: ensure a model exists or train one."""
        # Clean up existing model before testing to ensure fresh start for some tests
        if os.path.exists(ModelHandler.MODEL_FILE):
            os.remove(ModelHandler.MODEL_FILE)
        
        # This will train and save the model if it doesn't exist
        cls.model_handler = ModelHandler()
        # Verify model is loaded
        cls.assertTrue(cls.model_handler.model is not None, "Model should be loaded or trained.")

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests: remove the generated model file."""
        if os.path.exists(ModelHandler.MODEL_FILE):
            os.remove(ModelHandler.MODEL_FILE)
        
        # Clean up logs directory if it was created
        log_dir = 'logs'
        if os.path.exists(log_dir):
            for f in os.listdir(log_dir):
                os.remove(os.path.join(log_dir, f))
            os.rmdir(log_dir)

    def test_model_loading_or_training(self):
        """Test that the model is successfully loaded or trained."""
        self.assertIsNotNone(self.model_handler.model)
        self.assertTrue(os.path.exists(ModelHandler.MODEL_FILE))

    def test_predict_digit_basic(self):
        """Test prediction for a simple, clear digit (e.g., a white square for 0)."""
        # Create a blank white image (which the model might interpret as 0 or background)
        # For a more robust test, you'd load a known digit image.
        test_image_white = Image.new('L', (300, 300), 255) # White image

        # For a better test, let's draw a '1' in the center
        test_image_one = Image.new('L', (300, 300), 255)
        draw = ImageDraw.Draw(test_image_one)
        # Draw a vertical line for '1'
        center_x, center_y = 150, 150
        draw.line([(center_x, center_y - 50), (center_x, center_y + 50)], fill=0, width=20, joint="curve")

        predicted_digit, confidence = self.model_handler.predict_digit(test_image_one)
        
        # A '1' drawn like this should ideally be recognized with high confidence
        self.assertIn(predicted_digit, [1, 7], "Should predict a digit close to 1 or 7") # Sometimes '1' looks like '7'
        self.assertGreaterEqual(confidence, 0.5, "Confidence should be reasonably high for a clear drawing")

    def test_predict_digit_invalid_input(self):
        """Test prediction with an invalid image type."""
        with self.assertRaises(AttributeError): # PIL Image operations might raise this
            # Pass a numpy array directly without PIL Image conversion
            self.model_handler.predict_digit(np.random.rand(28, 28) * 255)

    def test_predict_digit_no_model(self):
        """Test prediction when model is not loaded (simulated)."""
        temp_model_handler = ModelHandler()
        temp_model_handler.model = None # Force no model loaded
        predicted_digit, confidence = temp_model_handler.predict_digit(Image.new('L', (300, 300), 0))
        self.assertEqual(predicted_digit, -1)
        self.assertEqual(confidence, 0.0)

# How to run these tests:
# Navigate to the 'digit_recognizer' directory in your terminal
# python -m unittest tests/test_model_handler.py
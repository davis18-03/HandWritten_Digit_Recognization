import logging
from model_handler import ModelHandler
from gui_app import DigitRecognizerApp
from utils import setup_logging

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

def main():
    """Main function to initialize and run the digit recognizer application."""
    logger.info("Starting Digit Recognizer application.")

    try:
        # Initialize ModelHandler (handles model loading/training)
        # This will ensure the model is ready before the GUI starts
        model_handler = ModelHandler()
        
        # Initialize and run the GUI application
        app = DigitRecognizerApp(model_handler)
        app.mainloop()
        
    except Exception as e:
        logger.critical(f"An unhandled error occurred: {e}", exc_info=True)
        # Optionally, show a messagebox for critical errors
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw() # Hide the main window
        messagebox.showerror("Critical Error", f"The application encountered a critical error and must close:\n{e}\nCheck logs for details.")
        root.destroy()
    finally:
        logger.info("Application shut down.")

if __name__ == "__main__":
    main()
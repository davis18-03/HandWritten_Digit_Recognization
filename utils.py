import logging
import os

def setup_logging(log_file='app_log.log', level=logging.INFO):
    """
    Sets up a robust logging configuration for the application.
    Messages will be printed to console and saved to a file.
    """
    # Create logs directory if it doesn't exist
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    # Configure the root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),  # Log to file
            logging.StreamHandler()         # Log to console
        ]
    )
    # Set a higher level for external libraries if needed to reduce verbosity
    logging.getLogger('tensorflow').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('keras').setLevel(logging.WARNING) # Keras logs through tensorflow as well
    logging.getLogger(__name__).info(f"Logging configured. Logs saved to {log_path}")
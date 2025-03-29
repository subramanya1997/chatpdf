import os
import logging
from logging.handlers import RotatingFileHandler
import sys

# Create logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

# Configure the root logger
def setup_logging():
    # Define log formatters
    # For file logs: detailed format with timestamp, logger name, level, file location, and message
    file_formatter = logging.Formatter(
        '%(asctime)s [%(name)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # For console logs: simplified format with timestamp, level, and message
    console_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Create handlers
    # File handler with rotation (10 MB per file, keep 5 backup files)
    file_handler = RotatingFileHandler(
        'logs/app.log',
        maxBytes=10*1024*1024,  # 10 MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Set specific levels for some loggers
    logging.getLogger('chainlit').setLevel(logging.INFO)
    logging.getLogger('openai').setLevel(logging.INFO)
    logging.getLogger('pinecone').setLevel(logging.INFO)

    # Log startup message
    root_logger.info('Logging system initialized') 
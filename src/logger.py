import logging
import os 
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
Logs_dir = os.path.join(os.getcwd(), 'logs')  # Define the directory path
os.makedirs(Logs_dir, exist_ok=True)  # Create the directory if it doesn't exist

LOG_FILE_PATH = os.path.join(Logs_dir, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

if __name__ == "__main__":
    logging.info("logging has started")


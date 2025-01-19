import logging
import sys

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("FSGenerator.log", mode="w"),  # Save logs to a file
        logging.StreamHandler(sys.stdout)  # Print logs to the console
    ]
)

# Create and configure the global logger
logger = logging.getLogger("MainLogger")
logger.info("Logger initialized with file and console handlers.")
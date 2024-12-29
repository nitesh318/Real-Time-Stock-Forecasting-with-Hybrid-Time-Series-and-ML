from kafka import KafkaConsumer
import json
import logging
import signal
import sys

# Kafka configuration
KAFKA_BROKER = "localhost:9092"
TOPIC_NAME = "stock_predictions"

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Kafka consumer
consumer = KafkaConsumer(
    TOPIC_NAME,
    bootstrap_servers=KAFKA_BROKER,
    auto_offset_reset='earliest',  # Start from the earliest message
    enable_auto_commit=True,
    group_id='stock-group',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)


def process_message(message):
    """Process the Kafka message."""
    logger.info(f"Processing message: {message}")
    # Add your processing logic here, e.g., storing in a database or triggering alerts


def signal_handler(signal, frame):
    """Handle graceful shutdown on keyboard interrupt."""
    logger.info("Gracefully shutting down...")
    consumer.close()
    sys.exit(0)


if __name__ == "__main__":
    # Register the signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)

    logger.info("Listening to Kafka topic...")

    try:
        for message in consumer:
            process_message(message.value)
    except Exception as e:
        logger.error(f"Error while consuming message: {e}")
    finally:
        logger.info("Consumer has been closed.")

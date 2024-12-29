import json
import logging
import os

from kafka import KafkaProducer
from kafka.errors import KafkaError

# Kafka configuration
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
TOPIC_NAME = "stock_predictions"

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Kafka producer
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    acks='all',  # Wait for all brokers to acknowledge the message
    retries=5  # Retry sending the message up to 5 times if it fails
)


def send_prediction_to_kafka(company_id, predictions):
    """Send stock prediction to Kafka."""
    message = {
        "company_id": company_id,
        "predictions": predictions
    }

    try:
        # Send message asynchronously
        future = producer.send(TOPIC_NAME, value=message)

        # Ensure the message is sent by waiting for the result (optional)
        record_metadata = future.get(timeout=10)  # You can adjust timeout based on your needs

        # Log success
        logger.info(
            f"Message sent to Kafka: {message}, partition: {record_metadata.partition}, offset: {record_metadata.offset}"
        )

    except KafkaError as e:
        logger.error(f"Kafka error while sending message to Kafka: {e}")
    except Exception as e:
        logger.error(f"Error sending message to Kafka: {e}")

    finally:
        # Flush producer to ensure messages are sent
        producer.flush()
        logger.info("Producer flushed and ready for next message.")

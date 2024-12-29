import os

from flask import Flask, jsonify
from kafka import KafkaProducer
from pymongo import MongoClient

app = Flask(__name__)

# MongoDB configuration
mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(mongo_uri)
try:
    client.admin.command("ping")
    mongo_status = "Connected"
except Exception as e:
    mongo_status = f"Error: {str(e)}"

# Kafka configuration
kafka_bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
try:
    producer = KafkaProducer(bootstrap_servers=kafka_bootstrap_servers)
    kafka_status = "Connected"
except Exception as e:
    kafka_status = f"Error: {str(e)}"


@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    """Healthcheck endpoint."""
    return jsonify({
        "flask_status": "Running",
        "mongo_status": mongo_status,
        "kafka_status": kafka_status
    }), 200


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)

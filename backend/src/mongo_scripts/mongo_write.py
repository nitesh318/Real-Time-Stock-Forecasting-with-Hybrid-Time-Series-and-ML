import logging

from pymongo import MongoClient


class MongoWrite:
    _client = None  # Reuse the MongoDB client for efficiency

    def __init__(self, mongo_uri, database_name):
        if MongoWrite._client is None:
            try:
                MongoWrite._client = MongoClient(mongo_uri)
                self.db = MongoWrite._client[database_name]
            except Exception as e:
                logging.error(f"Error connecting to MongoDB: {e}")
                raise
        else:
            self.db = MongoWrite._client[database_name]

    def insert_prediction(self, collection, data):
        try:
            # Check if the prediction already exists (based on company_id and date)
            if not self.db[collection].find_one({"company_id": data["company_id"], "date": data["date"]}):
                self.db[collection].insert_one(data)
                logging.info(f"Data inserted into {collection}: {data}")
            else:
                logging.info("Duplicate data found, skipping insertion.")
        except Exception as e:
            logging.error(f"Error inserting data into {collection}: {e}")

    def insert_predictions(self, collection, data_list):
        try:
            # Bulk insert for multiple predictions
            if data_list:
                self.db[collection].insert_many(data_list)
                logging.info(f"{len(data_list)} documents inserted into {collection}")
            else:
                logging.warning("No data to insert.")
        except Exception as e:
            logging.error(f"Error inserting multiple documents into {collection}: {e}")

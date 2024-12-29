import logging

from pymongo import MongoClient, ASCENDING


class MongoRead:
    _client = None  # Reuse the MongoDB client for efficiency

    def __init__(self, mongo_uri, database_name):
        if MongoRead._client is None:
            try:
                MongoRead._client = MongoClient(mongo_uri)
                self.db = MongoRead._client[database_name]
            except Exception as e:
                logging.error(f"Error connecting to MongoDB: {e}")
                raise
        else:
            self.db = MongoRead._client[database_name]

        # Ensure the `date` field is indexed for better query performance
        self.db.StockSeries.create_index([('date', ASCENDING)])

    def get_all_companies(self):
        try:
            return list(self.db.Companies.find({}, {'_id': 0}))
        except Exception as e:
            logging.error(f"Error retrieving companies: {e}")
            return []

    def get_stock_series_for_last_month(self, company_id, date):
        try:
            # Ensure `company_id` and `date` are indexed for faster queries
            self.db.StockSeries.create_index([('company_id', ASCENDING), ('date', ASCENDING)])
            results = list(self.db.StockSeries.find({"company_id": company_id, "date": {"$lt": date}}))

            # Convert `date` field if needed (e.g., string to datetime)
            for result in results:
                result['date'] = result.get('date')  # Ensure proper date format if necessary
            return results
        except Exception as e:
            logging.error(f"Error retrieving stock series: {e}")
            return []

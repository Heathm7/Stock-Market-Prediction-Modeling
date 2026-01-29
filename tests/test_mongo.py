from pymongo import MongoClient

# Replace with your MongoDB URI
DB_URI = "mongodb+srv://Heathm7:uTuzdXKHNfoqYNL8@stockprediction.34qeeah.mongodb.net/?appName=StockPrediction"
DB_NAME = "stock_prediction"
COLLECTION_NAME = "market_data"

# Connect to MongoDB
client = MongoClient(DB_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

print("Connected to MongoDB!")

# Show how many documents are already in the collection
print("Number of documents in collection:", collection.count_documents({}))

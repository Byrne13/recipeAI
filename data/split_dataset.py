from pymongo import MongoClient
import pandas as pd
from sklearn.model_selection import train_test_split

# Connect to the MongoDB server
client = MongoClient('mongodb://127.0.0.1:27017/')
db = client['JBrecipegen']  # Replace with your database name
collection = db['recipes']

# Get all the data from the database
cursor = collection.find()
all_data = pd.DataFrame(list(cursor))

# Save the data to a pickle file
all_data.to_pickle('all_data.pkl')

# Split the data into 80% validation and 20% testing
val_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42)

# Save the split data to a pickle file
val_data.to_pickle('val_data.pkl')
test_data.to_pickle('test_data.pkl')

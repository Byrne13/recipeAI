from pymongo import MongoClient

import pandas as pd

from sklearn.model_selection import train_test_split

client = MongoClient('mongodb://127.0.0.1:27017/')
db = client['JBrecipegen']  # Replace with your database name
collection = db['recipes'] 

cursor = collection.find()
data = pd.DataFrame(list(cursor))



# Split the data into 80% training and 20% testing
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Further split the training data into 80% training and 20% validation
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

train_data.to_pickle('train_data.pkl')
val_data.to_pickle('val_data.pkl')
test_data.to_pickle('test_data.pkl')
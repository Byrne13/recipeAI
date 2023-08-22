import sys
sys.path.append('..') # Adds the parent directory to the path

from recipe_recommender import generate_new_recipe, preprocess_data

import pickle

# relative file path
working_data_file = '../data/test_data.pkl'

# Load the test data
with open(working_data_file, 'rb') as file:
    test_data = pickle.load(file)

preprocessed_data, classes, mlb = preprocess_data(test_data)

# Usage example
#user_ingredients = ["tomato", "basil", "garlic"]
user_ingredients = ["cheese"]
new_recipe = generate_new_recipe(user_ingredients, test_data, preprocessed_data, mlb)
print(new_recipe)
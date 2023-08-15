from sklearn.metrics.pairwise import cosine_similarity

from sklearn.preprocessing import MultiLabelBinarizer

import numpy as np


import re

def preprocess_data(data):
    processed_ingredients = []

    # Iterate through the DataFrame's rows
    for index, row in data.iterrows():
        # Access the 'ingredients' column using the row's index
        ingredients = row['ingredients']
        parsed_ingredients = [re.findall(r'\b\w+\b', ingredient)[-1] for ingredient in ingredients if ingredient.strip()]
        processed_ingredients.append(parsed_ingredients)

    # Initialize the MultiLabelBinarizer
    mlb = MultiLabelBinarizer()

    # Fit the MultiLabelBinarizer to the ingredients list and transform the data
    processed_data = mlb.fit_transform(processed_ingredients)

    return processed_data.tolist(), mlb.classes_, mlb


def compute_cosine_similarity(user_vector, all_recipes):
    user_vector = np.array(user_vector).reshape(1, -1)

    # Transform the list of lists into a numpy array
    all_recipes = np.array(all_recipes)

    similarities = [cosine_similarity(user_vector, recipe.reshape(1, -1))[0][0] for recipe in all_recipes]

    return similarities


def generate_new_recipe(user_ingredients, original_data, all_recipes, mlb, top_n=5):
    user_vector = mlb.transform([user_ingredients])

    # Compute similarities using the updated function
    similarities = compute_cosine_similarity(user_vector, all_recipes)

    # Get top N similar recipes
    top_indexes = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top_n]
    top_recipes = [original_data.iloc[i] for i in top_indexes]

    new_recipe_name = "New Recipe Featuring " + ", ".join(user_ingredients)
    new_recipe_ingredients = []
    for recipe in top_recipes:
        new_recipe_ingredients += recipe['ingredients']

    new_recipe = {
        'name': new_recipe_name,
        'ingredients': new_recipe_ingredients
    }

    return new_recipe



from sklearn.metrics.pairwise import cosine_similarity

from sklearn.preprocessing import MultiLabelBinarizer

from keras.preprocessing.text import Tokenizer 

from keras.preprocessing.sequence import pad_sequences

import numpy as np

import re



def preprocess_instructions(data, max_instruction_length=None):
    # Process instructions at the recipe level
    all_instructions = [' '.join(instructions) for instructions in data['instructions']]

    # Initialize the tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_instructions)

    # Tokenize and pad the instructions
    tokenized_instructions = tokenizer.texts_to_sequences(all_instructions)

    # Pad the tokenized instructions
    tokenized_instructions = pad_sequences(tokenized_instructions, maxlen=max_instruction_length)

    return np.array(tokenized_instructions), tokenizer


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

    return np.array(processed_data.tolist()), mlb.classes_, mlb


def compute_cosine_similarity(user_vector, all_recipes):
    user_vector = np.array(user_vector).reshape(1, -1)

    # Transform the list of lists into a numpy array
    all_recipes = np.array(all_recipes)

    similarities = [cosine_similarity(user_vector, recipe.reshape(1, -1))[0][0] for recipe in all_recipes]

    return similarities

def create_instruction_targets(tokenized_instructions, max_instruction_length, vocabulary_size):
    target_data = [seq[1:] for seq in tokenized_instructions]  # Shift by one timestep

    # Initialize a 3D zero array
    instruction_target_data = np.zeros(
        (len(target_data), max_instruction_length, vocabulary_size), dtype="float32"
    )

    # Iterate and one-hot encode
    for i, sequence in enumerate(target_data):
        for t, word_index in enumerate(sequence):
            if t < max_instruction_length - 1:
                instruction_target_data[i, t, word_index] = 1.0

    return instruction_target_data


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



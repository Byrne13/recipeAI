from sklearn.metrics.pairwise import cosine_similarity

from sklearn.preprocessing import MultiLabelBinarizer

# Preprocess the data
def preprocess_data(data):
    # Extract the 'ingredients' field from each recipe
    #ingredients_list = [recipe['ingredients'] for recipe in data]

    # Initialize the MultiLabelBinarizer
    mlb = MultiLabelBinarizer()

    # Fit the MultiLabelBinarizer to the ingredients list and transform the data
    processed_data = mlb.fit_transform(data)
    
    return processed_data.tolist()

# Calculate similarity between two recipes
def calculate_similarity(recipe1, recipe2):
    return cosine_similarity(recipe1, recipe2)

# Recommend similar recipes
def recommend_recipes(target_recipe, all_recipes, top_n=5):
    similarities = [calculate_similarity(target_recipe, recipe) for recipe in all_recipes]
    return sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top_n]

# Example usage
#processed_data = preprocess_data(data)
#recommended_recipes = recommend_recipes(target_recipe, processed_data)


from recipe_recommender import preprocess_data

import pickle


# Load the test data
with open('test_data.pkl', 'rb') as file:
    test_data = pickle.load(file)

# Print the type of test_data
# print('type 1 ' + str(type(test_data)))

# Preprocess the data
preprocessed_data, classes = preprocess_data(test_data)
print(preprocessed_data)


# Print the matrix along with the recipe name
for processed_row, (index, recipe_row) in zip(preprocessed_data, test_data.iterrows()):
    print("Recipe:", recipe_row['title'])
    print(processed_row)


# Print the key
print("Key:")
for i, ingredient in enumerate(classes):
    print(f"{i}: {ingredient}")


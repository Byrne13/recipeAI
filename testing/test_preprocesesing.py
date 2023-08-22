import sys
sys.path.append('..') # Adds the parent directory to the path

from recipe_recommender import preprocess_data, preprocess_instructions # Added import
import pickle


# relative file path
working_data_file = '../data/test_data.pkl'

# Load the test data
with open(working_data_file, 'rb') as file:
    test_data = pickle.load(file)

# Preprocess the ingredients data
preprocessed_data, classes, mlb = preprocess_data(test_data)
print(preprocessed_data)

# Preprocess the instructions data
tokenized_instructions, tokenizer = preprocess_instructions(test_data)
print(tokenized_instructions) # Print the tokenized instructions

# Test the tokenizer
index_to_word = {v: k for k, v in tokenizer.word_index.items()}

# Function to convert tokenized sequence back to words
def tokens_to_words(tokenized_sequence):
    return ' '.join([index_to_word[token] for token in tokenized_sequence])

tokenized_instruction = [5, 2, 15]
original_instruction = tokens_to_words(tokenized_instruction)
print(original_instruction) 
# Output: "with the sugar"


# Print the matrix along with the recipe name
#for processed_row, (index, recipe_row) in zip(preprocessed_data, test_data.iterrows()):
#    print("Recipe:", recipe_row['title'])
#    print(processed_row)

# Print the key
#print("Key:")
#for i, ingredient in enumerate(classes):
#    print(f"{i}: {ingredient}")

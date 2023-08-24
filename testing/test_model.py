import tensorflow as tf
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
import sys

sys.path.append('..') # Adds the parent directory to the path
from recipe_recommender import preprocess_data, preprocess_instructions


# Path to saved model
model_path_tf = "../saved_model/my_model"

# Load the model
model = tf.keras.models.load_model(model_path_tf)
print(f"Model loaded from {model_path_tf}")

# file paths
data_file = '../data/test_data.pkl'
classes_file = '../data/classes.pkl'
mlb_file = '../data/mlb.pkl'
max_instruction_length_file = '../data/max_instruction_length.pkl'
num_ingredient_features_file = '../data/num_ingredient_features.pkl'

# Load the test data
with open(data_file, 'rb') as file:
    data = pickle.load(file)

# Load the tokenizer
with open('../data/tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

# load the MultiLabelBinarizer
with open(mlb_file, 'rb') as file:
    mlb = pickle.load(file)

# load the max_instruction_length
with open(max_instruction_length_file, 'rb') as file:
    max_instruction_length = pickle.load(file)

# load the num_ingredient_features
with open(num_ingredient_features_file, 'rb') as file:
    num_ingredient_features = pickle.load(file)

# Preprocess Ingredients using the same method as during training
processed_ingredients, _, _ = preprocess_data(data, mlb, num_ingredient_features)

# Now the shape of processed_ingredients should match what the model expects
print(f"Processed ingredients shape: {processed_ingredients.shape}")

# Preprocess instructions
tokenized_instructions = preprocess_instructions(data, tokenizer, max_instruction_length)


# Extract the existing layers
existing_layers = model.layers[:-1]

# Create a new input layer
new_input = Input(shape=model.input_shape[1:])

# Reconstruct the existing architecture
x = new_input
for layer in existing_layers:
    x = layer(x)

# Add your new dense layer
x = tf.keras.layers.Dense(371, activation='sigmoid')(x)

# Create a new model with the existing input and new output
new_model = tf.keras.models.Model(inputs=new_input, outputs=x)

# You can copy the weights from the old model to the new model if needed
new_model.set_weights(model.get_weights())

predictions = new_model.predict([processed_ingredients, tokenized_instructions])

print("Model's output shape:", model.output_shape)
print("Number of classes in mlb:", len(mlb.classes_))


# Define a threshold
threshold = 0.5

# Convert the probabilities to binary values based on the threshold
binary_predictions = (predictions >= threshold).astype(int)

# Use the MultiLabelBinarizer to convert the binary values to class labels
predicted_labels = mlb.inverse_transform(binary_predictions)

# Optionally, you can format the labels into strings if needed
formatted_predictions = [' '.join(labels) for labels in predicted_labels]

print("Predictions made on test data:")
print(predicted_labels[:5])  # Print the first 5 predictions


#print(formatted_predictions)
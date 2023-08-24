import tensorflow as tf
import pickle
import numpy as np
import re

from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
from keras.layers import Concatenate
from keras.preprocessing.text import Tokenizer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from recipe_recommender import preprocess_data, preprocess_instructions, create_instruction_targets 

all_data_file = './data/all_data.pkl'
val_data_file = './data/val_data.pkl'

tokenizer_file = './data/tokenizer.pkl'
classes_file = './data/classes.pkl'
mlb_file = './data/mlb.pkl'
max_instruction_length_file = './data/max_instruction_length.pkl'
num_ingredient_features_file = './data/num_ingredient_features.pkl'
model_path_tf = "saved_model/my_model"

batch_size = 64
epochs = 4
latent_dim = 256
embedding_dim = 100


# CREATE AN MLB OBJECT FIT TO ALL DATA
# Load all data to fit the MultiLabelBinarizer
with open(all_data_file, 'rb') as file:
    all_data = pickle.load(file)

# Initialize the MultiLabelBinarizer
mlb = MultiLabelBinarizer()
all_processed_ingredients, classes, mlb = preprocess_data(all_data, mlb, None)
num_ingredient_features = all_processed_ingredients.shape[1]

# Save the number of ingredient features to a file
with open(num_ingredient_features_file, 'wb') as file:
    pickle.dump(num_ingredient_features, file)

# Save the MultiLabelBinarizer to a file
with open(mlb_file, 'wb') as file:
    pickle.dump(mlb, file)

# Save the classes to a file
with open(classes_file, 'wb') as file:
    pickle.dump(classes, file)


# Initialize the tokenizer
tokenizer = Tokenizer()
all_instructions = [' '.join(instructions) for instructions in all_data['instructions']]
tokenizer.fit_on_texts(all_instructions)

# Tokenize the instructions to calculate max_instruction_length
temp_tokenized_instructions = tokenizer.texts_to_sequences(all_instructions)
max_instruction_length = max([len(seq) for seq in temp_tokenized_instructions])

# Save the max_instruction_length to a file
with open(max_instruction_length_file, 'wb') as file:
    pickle.dump(max_instruction_length, file)

print(f"Max instruction length: {max_instruction_length}")

# Save the tokenizer to a file
with open(tokenizer_file, 'wb') as file:
    pickle.dump(tokenizer, file)


# PREPROCESS AND SPLIT THE VALIDATION DATA
# Load the test data
with open(val_data_file, 'rb') as file:
    val_data = pickle.load(file)

# Preprocess Ingredients
processed_data, _, _ = preprocess_data(val_data, mlb, num_ingredient_features) 

# Preprocess Instructions
tokenized_instructions = preprocess_instructions(val_data, tokenizer, max_instruction_length)

# Define training parameters
vocabulary_size = len(tokenizer.word_index) + 1

print(f"Number of ingredient features: {num_ingredient_features}")

# Split into Training and Validation Sets
train_prepr_data, val_prepr_data, train_tokenized_instr, val_tokenized_instr = train_test_split(
    processed_data, tokenized_instructions, test_size=0.2, random_state=42)

# Print shapes to confirm
print(f"Train preprocessed data shape: {train_prepr_data.shape}")
print(f"Validation preprocessed data shape: {val_prepr_data.shape}")

# Create instruction targets for both training and validation
train_instr_target_data = create_instruction_targets(train_tokenized_instr, max_instruction_length, vocabulary_size)
val_instr_target_data = create_instruction_targets(val_tokenized_instr, max_instruction_length, vocabulary_size) 

# Print the shapes after creating target data
print(f"Training instruction target data shape: {train_instr_target_data.shape}")
print(f"Validation instruction target data shape: {val_instr_target_data.shape}")


# DEFINE THE MODEL
# Define an input layer
ingredient_input = Input(shape=(num_ingredient_features,)) # Shape: (None, 592)
instruction_input = Input(shape=(max_instruction_length,))  # Shape: (None, 350)

# Define an embedding layer
instruction_embedding = Embedding(output_dim=embedding_dim, input_dim=vocabulary_size)(instruction_input) # Shape: (None, 350, 100)

# Project ingredient_input into a compatible shape
ingredient_dense = Dense(embedding_dim)(ingredient_input) # Shape: (None, 100)
ingredient_repeated = tf.keras.layers.RepeatVector(max_instruction_length)(ingredient_dense) # Shape: (None, 350, 100)

# Concatenate instruction_embedding and ingredient_repeated
combined_input = Concatenate(axis=-1)([instruction_embedding, ingredient_repeated]) # Shape: (None, 350, 200)

# Define a LSTM model to encode the instructions
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(combined_input) 
encoder_states = [state_h, state_c]

# Define a LSTM model to decode the instructions
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(instruction_embedding, initial_state=encoder_states)

# Define a Dense layer to output the final instruction probabilities
output_layer = Dense(vocabulary_size, activation='softmax')
decoder_outputs = output_layer(decoder_outputs)


# COMPILE AND FIT THE MODEL
# Compile the model
print("Compiling the model...")
model = Model([ingredient_input, instruction_input], decoder_outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model
print("Fitting the model...")
model.fit([train_prepr_data, train_tokenized_instr], train_instr_target_data, epochs=epochs, batch_size=batch_size, validation_data=([val_prepr_data, val_tokenized_instr], val_instr_target_data))

# Save the model
model.save(model_path_tf, save_format="tf")
print(f"Model saved to {model_path_tf}")



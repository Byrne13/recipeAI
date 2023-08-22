import tensorflow as tf
import pickle

from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
from keras.layers import Concatenate

from recipe_recommender import preprocess_data, preprocess_instructions, create_instruction_targets 


training_data_file = './data/train_data.pkl'
validation_data_file = './data/val_data.pkl'

batch_size = 64
epochs = 10
latent_dim = 256
embedding_dim = 100

# Load the test data
with open(training_data_file, 'rb') as file:
    train_data = pickle.load(file)

# Preprocess the data
train_prepr_data, classes, mlb = preprocess_data(train_data)
train_tokenized_instr, tokenizer = preprocess_instructions(train_data)

max_instruction_length = max([len(seq) for seq in train_tokenized_instr])
num_ingredient_features = len(classes)
vocabulary_size = len(tokenizer.word_index) + 1

# Print the shapes after preprocessing
print(f"Training preprocessed data length: {len(train_prepr_data)}")
print(f"Training tokenized instructions length: {len(train_tokenized_instr)}")
print(f"Max instruction length: {max_instruction_length}")
print(f"Num ingredient features: {num_ingredient_features}")
print(f"Vocabulary size: {vocabulary_size}")

# Create instruction targets
train_instr_target_data = create_instruction_targets(train_tokenized_instr, max_instruction_length, vocabulary_size)

# Print the shapes after creating target data
print(f"Training instruction target data shape: {train_instr_target_data.shape}")

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
encoder_outputs, state_h, state_c = encoder_lstm(combined_input) # Change from "ingredient_input" to "combined_input"
encoder_states = [state_h, state_c]

# Define a LSTM model to decode the instructions
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(instruction_embedding, initial_state=encoder_states)

# Define a Dense layer to output the final instruction probabilities
output_layer = Dense(vocabulary_size, activation='softmax')
decoder_outputs = output_layer(decoder_outputs)

# Compile the model
print("Compiling the model...")
model = Model([ingredient_input, instruction_input], decoder_outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load the validation data
with open(validation_data_file, 'rb') as file:
    validation_data = pickle.load(file)

# Preprocess the validation data and create instruction targets
val_prepr_data, classes2, mlb2 = preprocess_data(validation_data)
val_tokenized_instr, tokenizer2 = preprocess_instructions(validation_data)
val_instr_target_data = create_instruction_targets(val_tokenized_instr, max_instruction_length, vocabulary_size)

# Print the shapes after loading and preprocessing validation data
print(f"Validation preprocessed data length: {len(val_prepr_data)}")
print(f"Validation tokenized instructions shape: {len(val_tokenized_instr)}")
print(f"Validation instruction target data shape: {val_instr_target_data.shape}")

print("Train preprocessed data shape:", train_prepr_data.shape)
print("Validation preprocessed data shape:", val_prepr_data.shape)


# Fit the model
print("Fitting the model...")
model.fit([train_prepr_data, train_tokenized_instr], train_instr_target_data, epochs=epochs, batch_size=batch_size, validation_data=([val_prepr_data, val_tokenized_instr], val_instr_target_data))

# Save the model
model_path_tf = "saved_model/my_model"
model.save(model_path_tf, save_format="tf")
print(f"Model saved to {model_path_tf}")



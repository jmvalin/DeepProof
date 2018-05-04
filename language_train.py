#!/usr/bin/python3
'''Sequence to sequence grammar check.
'''
from __future__ import print_function

import math
from keras.models import Model
from keras.layers import Input, LSTM, CuDNNLSTM, Dense, Embedding, Reshape, Concatenate, Lambda, Conv1D
from keras import backend as K
import numpy as np
import h5py
import sys
import encoding

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.29
set_session(tf.Session(config=config))

embed_dim = 64
batch_size = 128  # Batch size for training.
epochs = 1  # Number of epochs to train for.
latent_dim = 128  # Latent dimensionality of the encoding space.

with h5py.File(sys.argv[1], 'r') as hf:
    output_text = hf['output'][:]
decoder_target_data = np.reshape(output_text, (output_text.shape[0], output_text.shape[1], 1))
decoder_input_data = np.zeros((output_text.shape[0], output_text.shape[1], 1), dtype='uint8')
decoder_input_data[:,1:,:] = decoder_target_data[:,:-1,:]
max_decoder_seq_length = output_text.shape[1]
num_encoder_tokens = len(encoding.char_list)

print("Number of sentences: ", output_text.shape[0])
print("Sentence length: ", output_text.shape[1])
print("Number of chars: ", num_encoder_tokens)

# Define an input sequence and process it.
reshape1 = Reshape((-1, embed_dim))
embed = Embedding(num_encoder_tokens, embed_dim)

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, 1))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = CuDNNLSTM(latent_dim, return_sequences=True, return_state=True)

dec_lstm_input = reshape1(embed(decoder_inputs))

decoder_outputs, _, _ = decoder_lstm(dec_lstm_input)
decoder_dense = Dense(num_encoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model(decoder_inputs, decoder_outputs)

# Run training
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
model.summary()
model.fit(decoder_input_data, decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
# Save model
model.save('language.h5')
#model.load_weights('s2s.h5')

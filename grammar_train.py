#!/usr/bin/python3
'''Sequence to sequence grammar check.
'''
from __future__ import print_function

import math
from keras.models import Model
from keras.layers import Input, LSTM, CuDNNLSTM, Dense, Embedding, Reshape, Concatenate, Lambda, Conv1D
from keras.optimizers import Adam
from keras import backend as K
import numpy as np
import h5py
import sys
import encoding

import deepproof_model

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.44
set_session(tf.Session(config=config))

batch_size = 128  # Batch size for training.
epochs = 1  # Number of epochs to train for.

encoder_model, decoder_model, model = deepproof_model.create(True)

with h5py.File(sys.argv[1], 'r') as hf:
    input_text = hf['input'][:]
    output_text = hf['output'][:]
#input_text = input_text[0:8000, :]
#output_text = output_text[0:8000, :]
input_data = np.reshape(input_text, (input_text.shape[0], input_text.shape[1], 1))
decoder_target_data = np.reshape(output_text, (output_text.shape[0], output_text.shape[1], 1))
decoder_input_data = np.zeros((input_text.shape[0], input_text.shape[1], 1), dtype='uint8')
decoder_input_data[:,1:,:] = decoder_target_data[:,:-1,:]
max_decoder_seq_length = input_text.shape[1]
num_encoder_tokens = len(encoding.char_list)

print("Number of sentences: ", input_text.shape[0])
print("Sentence length: ", input_text.shape[1])
print("Number of chars: ", num_encoder_tokens)

# Run training
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
#model.load_weights('proof7c.h5')
model.summary()
model.fit([input_data[:,:,0:1], decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
# Save model
model.save('proof7e.h5')
model.compile(optimizer=Adam(0.0003), loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
model.fit([input_data[:,:,0:1], decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
model.save('proof7e2.h5')
model.fit([input_data[:,:,0:1], decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
model.save('proof7e3.h5')
model.fit([input_data[:,:,0:1], decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
model.save('proof7e4.h5')



start = int(.9*input_text.shape[0])
for seq_index in range(start, start+1000):
    # Take one sequence (part of the training test)
    # for trying out decoding.
    input_seq = input_data[seq_index: seq_index + 1]
    decoded_sentence0 = deepproof_model.decode_sequence([encoder_model, decoder_model], input_seq)
    decoded_sentence = deepproof_model.beam_decode_sequence([encoder_model, decoder_model], input_seq)
    deepproof_model.decode_ground_truth([encoder_model, decoder_model], input_seq, output_text[seq_index,:])
    print('-')
    print('Input sentence:   ', encoding.decode_string(input_text[seq_index,:]))
    print('Decoded sentence0:', decoded_sentence0)
    print('Decoded sentence: ', decoded_sentence)
    print('Original sentence:', encoding.decode_string(output_text[seq_index,:]))

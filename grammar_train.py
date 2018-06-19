#!/usr/bin/python3
'''Sequence to sequence grammar check.
'''
from __future__ import print_function

import math
from keras.models import Model
from keras.layers import Input, LSTM, CuDNNLSTM, Dense, Embedding, Reshape, Concatenate, Lambda, Conv1D
from keras.optimizers import Adam
from keras.utils import Sequence
from keras.callbacks import ModelCheckpoint
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
epochs = 4  # Number of epochs to train for.

encoder_model, decoder_model, model = deepproof_model.create(True)

class GrammarSequence(Sequence):

    def __init__(self, x1_set, x2_set, y_set, batch_size, validation_split=0.2, test=False):
        if test:
            self.x1 = x1_set[int(1+x1_set.shape[0]*(1-validation_split)):,:,:]
            self.x2 = x2_set[int(1+x2_set.shape[0]*(1-validation_split)):,:,:]
            self.y = y_set[int(1+y_set.shape[0]*(1-validation_split)):,:,:]
        else:
            self.x1 = x1_set[:int(x1_set.shape[0]*(1-validation_split)),:,:]
            self.x2 = x2_set[:int(x2_set.shape[0]*(1-validation_split)),:,:]
            self.y = y_set[:int(y_set.shape[0]*(1-validation_split)),:,:]
        self.batch_size = batch_size
        self.pe = np.repeat(deepproof_model.position_matrix(300, deepproof_model.pe_dim), batch_size, axis=0)

    def __len__(self):
        return len(self.x1) // self.batch_size

    def __getitem__(self, idx):
        batch_x1 = self.x1[idx * self.batch_size:(idx + 1) * self.batch_size, :, :]
        batch_x2 = self.x2[idx * self.batch_size:(idx + 1) * self.batch_size, :, :]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size, :, :]
        #print(batch_x.shape)
        #print(batch_y.shape)

        return [self.pe, batch_x1, batch_x2], batch_y

input_text = None
output_text = None
for file in sys.argv[1:]:
    with h5py.File(file, 'r') as hf:
        if input_text is None:
            input_text = hf['input'][:]
            output_text = hf['output'][:]
        else:
            input_text = np.concatenate([input_text, hf['input'][:]])
            output_text = np.concatenate([output_text, hf['output'][:]])
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
gen_train = GrammarSequence(input_data[:,:,0:1], decoder_input_data, decoder_target_data, batch_size)
gen_test = GrammarSequence(input_data[:,:,0:1], decoder_input_data, decoder_target_data, batch_size, test=True)
checkpoint = ModelCheckpoint('proof8d_{epoch:02d}.h5')

model.fit_generator(gen_train,
          callbacks=[checkpoint],
          epochs=1,
          validation_data=gen_test)

model.compile(optimizer=Adam(0.0003), loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
model.fit_generator(gen_train,
          callbacks=[checkpoint],
          initial_epoch=1,
          epochs=epochs,
          validation_data=gen_test)


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

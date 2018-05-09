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

import deepproof_model

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.29
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
model.summary()
model.fit([input_data[:,:,0:1], decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
# Save model
model.save('s2s.h5')
#model.load_weights('s2s.h5')


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    encoder_outputs, state_h, state_c = encoder_model.predict(input_seq[:,:,0:1])
    states_value = [state_h, state_c]

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, 1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, :] = 0

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    foo=0
    prob = 0
    while foo < input_seq.shape[1] and not stop_condition:
        #target_seq[0, 0, 0] = input_seq[0, foo, 0]
        output_tokens, h, c = decoder_model.predict(
            [target_seq, encoder_outputs[:,foo:foo+1,:]] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = encoding.char_list[sampled_token_index]
        decoded_sentence += sampled_char
        prob += math.log(output_tokens[0, -1, sampled_token_index])
        # Exit condition: either hit max length
        # or find stop character.
        #if ((foo > 1 and sampled_token_index <= 1) or
        #   len(decoded_sentence) > max_decoder_seq_length):
        #    stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, 1))
        target_seq[0, 0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]
        foo = foo+1
    print(prob)
    return decoded_sentence

def decode_ground_truth(input_seq, output_seq):
    # Encode the input as state vectors.
    encoder_outputs, state_h, state_c = encoder_model.predict(input_seq[:,:,0:1])
    states_value = [state_h, state_c]

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, 1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, :] = 0

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    foo=0
    prob = 0
    while foo < input_seq.shape[1] and not stop_condition:
        #target_seq[0, 0, 0] = input_seq[0, foo, 0]
        output_tokens, h, c = decoder_model.predict(
            [target_seq, encoder_outputs[:,foo:foo+1,:]] + states_value)

        # Sample a token
        sampled_token_index = output_seq[foo]
        sampled_char = encoding.char_list[sampled_token_index]
        decoded_sentence += sampled_char
        prob += math.log(output_tokens[0, -1, sampled_token_index])
        # Exit condition: either hit max length
        # or find stop character.
        #if ((foo > 1 and sampled_token_index <= 1) or
        #   len(decoded_sentence) > max_decoder_seq_length):
        #    stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, 1))
        target_seq[0, 0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]
        foo = foo+1
    print(prob)
    return prob

def beam_decode_sequence(input_seq):
    # Encode the input as state vectors.
    B = 10
    encoder_outputs, state_h, state_c = encoder_model.predict(input_seq[:,:,0:1])

    in_nbest=[(0., '', np.array([[[0]]]), [state_h, state_c])]
    foo=0
    while foo < input_seq.shape[1]:
        out_nbest = []
        for prob, decoded_sentence, target_seq, states_value in in_nbest:
            output_tokens, h, c = decoder_model.predict(
                [target_seq, encoder_outputs[:,foo:foo+1,:]] + states_value)
            arg = np.argsort(output_tokens[0, -1, :])
            # Sample a token
            # Update states
            states_value = [h, c]
            for i in range(B):
                sampled_token_index = arg[-1-i]
                sampled_char = encoding.char_list[sampled_token_index]
                # Update the target sequence (of length 1).
                new_prob = prob + math.log(output_tokens[0, -1, sampled_token_index])
                target_seq = np.array([[[sampled_token_index]]])
                candidate = (new_prob, decoded_sentence + sampled_char, target_seq, states_value)
                if len(out_nbest) < B:
                    out_nbest.append(candidate)
                elif new_prob > out_nbest[-1][0]:
                    for j in range(len(out_nbest)):
                        if new_prob > out_nbest[j][0]:
                            out_nbest = out_nbest[:j] + [candidate] + out_nbest[j+1:]
                            break
        
        in_nbest = out_nbest
        # Exit condition: either hit max length
        # or find stop character.
        #if ((foo > 1 and sampled_token_index <= 1) or
        #   len(decoded_sentence) > max_decoder_seq_length):
        #    break

        foo = foo+1
    print(in_nbest[0][0])
    #print("total = ", sum([math.exp(i[0]) for i in in_nbest]))
    return in_nbest[0][1]

start = int(.9*input_text.shape[0])
for seq_index in range(start, start+1000):
    # Take one sequence (part of the training test)
    # for trying out decoding.
    input_seq = input_data[seq_index: seq_index + 1]
    decoded_sentence0 = decode_sequence(input_seq)
    decoded_sentence = beam_decode_sequence(input_seq)
    decode_ground_truth(input_seq, output_text[seq_index,:])
    print('-')
    print('Input sentence:   ', encoding.decode_string(input_text[seq_index,:]))
    print('Decoded sentence0:', decoded_sentence0)
    print('Decoded sentence: ', decoded_sentence)
    print('Original sentence:', encoding.decode_string(output_text[seq_index,:]))

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

encoder_model, decoder_model, model, lang_model = deepproof_model.create(False)

model.load_weights('proof2.h5')


for line in sys.stdin:
    line = line.rstrip()
    input_seq = encoding.encode_string(line, 300, 0)
    input_seq = np.reshape(input_seq, (1, input_seq.shape[0], 1))
    decoded_sentence0 = deepproof_model.decode_sequence([encoder_model, decoder_model], input_seq)
    decoded_sentence = deepproof_model.beam_decode_sequence([encoder_model, decoder_model], input_seq)
    print('-')
    print('Input sentence:   ', encoding.decode_string(input_seq[0, :, 0]))
    print('Decoded sentence0:', decoded_sentence0)
    print('Decoded sentence: ', decoded_sentence)

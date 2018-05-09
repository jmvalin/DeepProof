import math
from keras.models import Model
from keras.layers import Input, LSTM, CuDNNLSTM, Dense, Embedding, Reshape, Concatenate, Lambda, Conv1D
from keras import backend as K
import numpy as np
import h5py
import sys
import encoding

embed_dim = 64
latent_dim = 512  # Latent dimensionality of the encoding space.
num_encoder_tokens = len(encoding.char_list)


def create(use_gpu):
    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None, 1))
    reshape1 = Reshape((-1, embed_dim))
    reshape2 = Reshape((-1, embed_dim))
    conv = Conv1D(latent_dim//2, 5, padding='same', activation='tanh')
    embed = Embedding(num_encoder_tokens, embed_dim)
    if use_gpu:
        encoder = CuDNNLSTM(latent_dim, return_sequences=True, return_state=True, go_backwards=True)
    else:
        encoder = LSTM(latent_dim, recurrent_activation="sigmoid", return_sequences=True, return_state=True, go_backwards=True)
    encoder_outputs, state_h, state_c = encoder(conv(reshape1(embed(encoder_inputs))))
    rev = Lambda(lambda x: K.reverse(x, 1))
    conv2 = Conv1D(latent_dim//2, 5, dilation_rate=2, padding='same', activation='tanh')

    encoder_outputs = conv2(rev(encoder_outputs))
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, 1))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    if use_gpu:
        decoder_lstm = CuDNNLSTM(latent_dim, return_sequences=True, return_state=True)
    else:
        decoder_lstm = LSTM(latent_dim, recurrent_activation="sigmoid", return_sequences=True, return_state=True)

    dec_lstm_input = reshape1(embed(decoder_inputs))
    dec_lstm_input = Concatenate()([dec_lstm_input, encoder_outputs])

    decoder_outputs, _, _ = decoder_lstm(dec_lstm_input,
                                         initial_state=encoder_states)
    decoder_dense = Dense(num_encoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    encoder_model = Model(encoder_inputs, [encoder_outputs, state_h, state_c])

    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_enc_inputs = Input(shape=(None, latent_dim//2))
    decoder_outputs, state_h, state_c = decoder_lstm(
        Concatenate()([reshape1(embed(decoder_inputs)), decoder_enc_inputs]), initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs, decoder_enc_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)
    return (encoder_model, decoder_model, model)

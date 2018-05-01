#!/usr/bin/python3

import numpy as np
import encoding
import h5py
import sys


with h5py.File(sys.argv[1], 'r') as hf:
    input_text = hf['input'][:]
    output_text = hf['output'][:]

maxlen = input_text.shape[1]
print(maxlen)
for i in range(input_text.shape[0]):
    line = input_text[i,:]
    orig = output_text[i,:]
    #print (encoding.decode_string(line))
    print(encoding.decode_string(line))
    print(encoding.decode_string(orig), '\n')
